import logging
import shelve
import pysam
import collections
import os
import numpy as np
import pickle
import tensorflow as tf
import sys

from spliceai.utils import get_cov, get_wid, get_seq, is_record_valid, is_location_predictable, \
        is_valid_alt_record, encode_seqs, create_unhandled_delta_score, get_alt_gene_delta_score


logger = logging.getLogger(__name__)


## CUSTOM DATA TYPES
SequenceType_REF = 0
SequenceType_ALT = 1

BatchLookupIndex = collections.namedtuple(
    #                    ref/alt       size        batch for this size    index in current batch for this size
    'BatchLookupIndex', 'sequence_type tensor_size batch_ix batch_index'
)

PreparedVCFRecord = collections.namedtuple(
    'PreparedVCFRecord', 'vcf_idx gene_info locations'
)


# class to parse input and prep batches
class VCFReader:
    def __init__(self, ann, input_data, prediction_batch_size, prediction_queue, tmpdir, dist):
        self.ann = ann
        # This is the maximum number of predictions to parse/encode/predict at a time
        self.prediction_batch_size = prediction_batch_size
        # the vcf file
        self.input_data = input_data
        # window to consider
        self.dist = dist
        # Batch vars
        self.batches = {}

        # Counts
        self.total_predictions = 0
        self.total_vcf_records = 0
        self.batch_counters = {}
        
        # the queue
        self.prediction_queue = prediction_queue

        # shelves to track data. 
        self.tmpdir = tmpdir 
        # track records to have order correct
        logging.debug("Opening spliceai_records shelf")
        try:
            self.shelf_records = shelve.open(os.path.join(self.tmpdir,"spliceai_records.shelf"))
        except Exception as e:
            logging.error(f"Could not open shelf: {e}")
            raise(e)


    def add_records(self):
        
        try:
            vcf = pysam.VariantFile(self.input_data)
        except (IOError, ValueError) as e:
            logging.error('{}'.format(e))
            raise(e)
        for record in vcf:
            try:
                self.add_record(record)
            except Exception as e:
                raise(e)
        vcf.close()
        

    def add_record(self, record):
        """
        Adds a record to a batch. It'll capture the gene information for the record and
        save it for later to avoid looking it up again, then it'll encode ref and alt from
        the VCF record and place the encoded values into lists of matching sizes. Once the
        encoded values are added, a BatchLookupIndex is created so that after the predictions
        are made, it knows where to look up the corresponding prediction for the vcf record.

        Once the batch size hits it's capacity, it'll process all the predictions for the
        encoded batch.
        """

        self.total_vcf_records += 1
        # Collect gene information for this record
        gene_info = self.ann.get_name_and_strand(record.chrom, record.pos)

        # Keep track of how many predictions we're going to make
        prediction_count = len(record.alts) * len(gene_info.genes)
        self.total_predictions += prediction_count

        # Collect lists of encoded ref/alt sequences
        x_ref, x_alt = self._encode_batch_records(record, self.ann, self.dist, gene_info)

        # List of BatchLookupIndex's so we know how to lookup predictions for records from
        # the batches
        batch_lookup_indexes = []

        # Process the encodings into batches
        for var_type, encoded_seq in zip((SequenceType_REF, SequenceType_ALT), (x_ref, x_alt)):

            if len(encoded_seq) == 0:
                # Add BatchLookupIndex with zeros so when the batch collects the outputs
                # it knows that there is no prediction for this record
                batch_lookup_indexes.append(BatchLookupIndex(var_type, 0, 0, 0))
                continue

            # Iterate over the encoded sequence and drop into the correct batch by size and
            # create an index to use to pull out the result after batch is processed
            for row in encoded_seq:
                # Extract the size of the sequence that was encoded to build a batch from
                tensor_size = row.shape[1]

                # Create batch for this size
                if tensor_size not in self.batches:
                    self.batches[tensor_size] = []
                    self.batch_counters[tensor_size] = 0

                # Add encoded record to batch 'n' for tensor_size 
                self.batches[tensor_size].append(row)

                # Get the index of the record we just added in the batch
                cur_batch_record_ix = len(self.batches[tensor_size]) - 1

                # Store a reference so we can pull out the prediction for this item from the batches
                batch_lookup_indexes.append(BatchLookupIndex(var_type, tensor_size, self.batch_counters[tensor_size] , cur_batch_record_ix))

        # Save the batch locations for this record on the composite object
        prepared_record = PreparedVCFRecord(
            vcf_idx=self.total_vcf_records, gene_info=gene_info, locations=batch_lookup_indexes
        )
        #  add to shelf by vcf_idx
        self.shelf_records[str(self.total_vcf_records)] = prepared_record

        # If we're reached our threshold for the max items to process, then process the batch
        for tensor_size in self.batch_counters:
            if len(self.batches[tensor_size]) >= self.prediction_batch_size:
                logger.debug("Batch {} full. Adding to queue".format(tensor_size))
                # fully prep the batch outside of gpu routine...
                data = np.concatenate(self.batches[tensor_size])
                concat_len = len(data)
                # offload conversion of batch from np to tensor to CPU
                with tf.device('CPU:0'):
                    data = tf.convert_to_tensor(data)
                queue_item = {'tensor_size': tensor_size, 'batch_ix': self.batch_counters[tensor_size], 'data' : data, 'length':concat_len}
                with open(os.path.join(self.tmpdir,"{}--{}.in.pickle".format(tensor_size,self.batch_counters[tensor_size])),"wb") as p:
                    pickle.dump(queue_item,p)
                self.prediction_queue.put("{}--{}.in.pickle".format(tensor_size,self.batch_counters[tensor_size]))
                
                # reset
                self.batches[tensor_size] = []
                self.batch_counters[tensor_size] += 1

        


    def finish(self,nr_workers):
        """
        Method to process all the remaining items that have been added to the batches.
        """
        logger.debug("Queueing remaining batches")
        for tensor_size in self.batch_counters:
                if len(self.batches[tensor_size] ) > 0:
                    # fully prep the batch outside of gpu routine...
                    data = np.concatenate(self.batches[tensor_size])
                    concat_len = len(data)
                    # offload conversion of batch from np to tensor to CPU
                    with tf.device('CPU:0'):
                        data = tf.convert_to_tensor(data)
                    queue_item = {'tensor_size': tensor_size, 'batch_ix': self.batch_counters[tensor_size], 'data' : data, 'length':concat_len}
                    with open(os.path.join(self.tmpdir,"{}--{}.in.pickle".format(tensor_size,self.batch_counters[tensor_size])),"wb") as p:
                         pickle.dump(queue_item,p)
                    self.prediction_queue.put("{}--{}.in.pickle".format(tensor_size,self.batch_counters[tensor_size]))
                    # clear
                    self.batches[tensor_size] = []
        # all done : push finish signals (one per process device..).
        logging.debug("Queueing finish signals")
        for i in range(nr_workers):
            self.prediction_queue.put('Finished')


    # Heavily based on utils.get_delta_scores but only handles the validation and encoding
    # of the record, but doesn't do any of the prediction or post-processing steps
    def _encode_batch_records(self, record, ann, dist_var, gene_info):
        cov = get_cov(dist_var)
        wid = get_wid(cov)
        # If the record is not going to get a prediction, return this empty encoding
        empty_encoding = ([], [])

        if not is_record_valid(record):
            return empty_encoding

        seq = get_seq(record, ann, wid)
        if not seq:
            return empty_encoding

        if not is_location_predictable(record, seq, wid, dist_var):
            return empty_encoding

        all_x_ref = []
        all_x_alt = []
        for alt_ix in range(len(record.alts)):
            for gene_ix in range(len(gene_info.idxs)):

                if not is_valid_alt_record(record, alt_ix):
                    continue

                x_ref, x_alt = encode_seqs(record=record,
                                       seq=seq,
                                       ann=ann,
                                       gene_info=gene_info,
                                       gene_ix=gene_ix,
                                       alt_ix=alt_ix,
                                       wid=wid)

                all_x_ref.append(x_ref)
                all_x_alt.append(x_alt)

        return all_x_ref, all_x_alt




# class to parse input and prep batches
class VCFWriter:
    def __init__(self, args, tmpdir, devices, ann):
        
        self.args = args
        # the vcf file
        self.input_data = args.input_data
        self.output_data = args.output_data
        # window to consider
        self.dist = args.distance
        # used devices
        self.devices = [x.name for x in devices]
        # shelves to track data. 
        self.tmpdir = tmpdir 
        # track records to have order correct
        self.shelf_records = shelve.open(os.path.join(self.tmpdir,"spliceai_records.shelf"))
        # trackers
        self.total_records = 0
        self.total_predictions = 0
        # annotations. 
        self.ann = ann

    def process(self):
        # prepare the global pred_shelf
        self._aggregate_predictions()

        # open the files & update header:
        self.vcf_in = pysam.VariantFile(self.input_data)
        header = self.vcf_in.header
        header.add_line('##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAIv1.3.1 variant '
                'annotation. These include delta scores (DS) and delta positions (DP) for '
                'acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). '
                'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">')
        self.vcf_out = pysam.VariantFile(self.output_data,mode='w',header=header)

        # write the output vcf.
        self._write_records()

        # close shelves
        self.shelf_records.close()
        self.shelf_preds.close()

        # close output file.
        self.vcf_in.close()
        self.vcf_out.close()

    # aggregate shelves over the devices
    def _aggregate_predictions(self):
        logger.debug("Aggregating device shelves")
        self.shelf_preds_name = f"spliceai_preds.shelf"
        self.shelf_preds = shelve.open(os.path.join(self.tmpdir, self.shelf_preds_name))
        for device in self.devices:
            device_shelf_name = f"spliceai_preds.{device[1:].replace(':','_')}.shelf"
            device_shelf_preds = shelve.open(os.path.join(self.tmpdir, device_shelf_name))
            for x in device_shelf_preds:
                self.shelf_preds[x] = device_shelf_preds[x]
            device_shelf_preds.close()

    

    # wrapper to write out all shelved variants
    def _write_records(self):
        logger.debug("Writing output file")
        # parse vcf
        line_idx = 0
        batch = []
        last_batch_key = ''
        for record in self.vcf_in:
            line_idx += 1  
            # get prepared record by line_idx
            prepared_record = self.shelf_records[str(line_idx)]
            gene_info = prepared_record.gene_info
            # (REF + #ALT ) * #genes   (* 5 models)
            self.total_predictions += (1 + len(record.alts)) * len(gene_info.genes)
            
            all_y_ref = []
            all_y_alt = []

            # Each prediction in the batch is located and put into the correct y
            for location in prepared_record.locations:
                # No prediction here
                if location.tensor_size == 0:
                    if location.sequence_type == SequenceType_REF:
                        all_y_ref.append(None)
                    else:
                        all_y_alt.append(None)
                    continue
                
                # Extract the prediction from the batch into a list of predictions for this record
                # recycle the batch variable if key is the same.
                if not last_batch_key == "{}|{}".format(location.tensor_size,location.batch_ix):
                    last_batch_key = "{}|{}".format(location.tensor_size,location.batch_ix)
                    batch = self.shelf_preds[last_batch_key]
                    
                if location.sequence_type == SequenceType_REF:
                    all_y_ref.append(batch[[location.batch_index], :, :])
                else:
                    all_y_alt.append(batch[[location.batch_index], :, :])
            # get delta scores 
            delta_scores = self._extract_delta_scores(
                all_y_ref=all_y_ref,
                all_y_alt=all_y_alt,
                record=record,
                gene_info=gene_info,
            )

            # If there are predictions, write them to the VCF INFO section
            if len(delta_scores) > 0:
                record.info['SpliceAI'] = delta_scores

            self.vcf_out.write(record)
        # close shelf again
        self.total_vcf_records = line_idx


    # Heavily based on utils.get_delta_scores but only handles the post-processing steps after
    # the models have made the predictions
    def _extract_delta_scores(self, all_y_ref, all_y_alt, record,  gene_info):
        # variables:
        dist_var = self.dist
        ann = self.ann
        mask = self.args.mask

        cov = get_cov(dist_var)
        delta_scores = []
        pred_ix = 0
        for alt_ix in range(len(record.alts)):
            for gene_ix in range(len(gene_info.idxs)):

            
                # Pull prediction out of batch
                try:
                    y_ref = all_y_ref[pred_ix]
                    y_alt = all_y_alt[pred_ix]
                except IndexError:
                    logger.warn("No data for record below, alt_ix {} : gene_ix {} : pred_ix {}".format(alt_ix, gene_ix,pred_ix))
                    logger.warn(record)
                    continue
                except Exception as e:
                    logger.error("Predction error: {}".format(e))
                    logger.error(record)
                    raise e

                # No prediction here
                if y_ref is None or y_alt is None:
                    continue

                if not is_valid_alt_record(record, alt_ix):
                    continue

                if len(record.ref) > 1 and len(record.alts[alt_ix]) > 1:
                    pred_ix += 1
                    delta_score = create_unhandled_delta_score(record.alts[alt_ix], gene_info.genes[gene_ix])
                    delta_scores.append(delta_score)
                    continue

                if pred_ix >= len(all_y_ref) or pred_ix >= len(all_y_alt):
                    raise LookupError(
                        'Prediction index {} does not exist in prediction matrices: ref({}) alt({})'.format(
                            pred_ix, len(all_y_ref), len(all_y_alt)
                        )
                    )

                delta_score = get_alt_gene_delta_score(record=record,
                                                   ann=ann,
                                                   alt_ix=alt_ix,
                                                   gene_ix=gene_ix,
                                                   y_ref=y_ref,
                                                   y_alt=y_alt,
                                                   cov=cov,
                                                   gene_info=gene_info,
                                                   mask=mask)
                delta_scores.append(delta_score)

                pred_ix += 1

        return delta_scores




