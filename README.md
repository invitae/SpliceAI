## SpliceAI: A deep learning-based tool to identify splice variants
[![release](https://img.shields.io/badge/release-v1.3.1-orange.svg)](https://img.shields.io/badge/release-v1.3.1-orange.svg)
[![license](https://img.shields.io/badge/license-GPLv3-green.svg)](https://img.shields.io/badge/license-GPLv3-green.svg)
[![downloads](https://pepy.tech/badge/spliceai)](https://pepy.tech/badge/spliceai)

This package annotates genetic variants with their predicted effect on splicing, as described in [Jaganathan *et al*, Cell 2019 in press](https://doi.org/10.1016/j.cell.2018.12.015). The annotations for all possible substitutions, 1 base insertions, and 1-4 base deletions within genes are available [here](https://basespace.illumina.com/s/otSPW8hnhaZR) for download. These annotations are free for academic and not-for-profit use; other use requires a commercial license from Illumina, Inc.

### License
SpliceAI source code is provided under the [GPLv3 license](LICENSE). SpliceAI includes several third party packages provided under other open source licenses, please see [NOTICE](NOTICE) for additional details. The trained models used by SpliceAI (located in this package at spliceai/models) are provided under the [CC BY NC 4.0](LICENSE) license for academic and non-commercial use; other use requires a commercial license from Illumina, Inc.

### Installation

This release can most easily be used as a docker container: 

```sh
docker pull cmgantwerpen/spliceai_v1.3:latest

docker run --gpus all cmgantwerpen/spliceai_v1.3:latest spliceai -h 
```

A container including reference and annotation data is available as well:


```sh
docker pull cmgantwerpen/spliceai_v1.3:full
```
Note that this version has a larger footprint (12Gb). Data is available for Genome Build hg19 and hg38 under /data/



The simplest way to install (the original version of) SpliceAI is through pip or conda:
```sh
pip install spliceai
# or
conda install -c bioconda spliceai
```

Alternately, SpliceAI can be installed from the [github repository](https://github.com/invitae/SpliceAI.git):
```sh
git clone https://github.com/invitae/SpliceAI.git
cd SpliceAI
python setup.py install
```

SpliceAI requires ```tensorflow>=1.2.0```, which is best installed separately via pip or conda (see the [TensorFlow](https://www.tensorflow.org/) website for other installation options):
```sh
pip install tensorflow
# or
conda install tensorflow
```

### Usage
SpliceAI can be run from the command line:
```sh
spliceai -I input.vcf -O output.vcf -R genome.fa -A grch37
# or you can pipe the input and output VCFs
cat input.vcf | spliceai -R genome.fa -A grch37 > output.vcf
```

Required parameters:
 - ```-I```: Input VCF with variants of interest.
 - ```-O```: Output VCF with SpliceAI predictions `ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL` included in the INFO column (see table below for details). Only SNVs and simple INDELs (REF or ALT is a single base) within genes are annotated. Variants in multiple genes have separate predictions for each gene.
 - ```-R```: Reference genome fasta file. Can be downloaded from [GRCh37/hg19](http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz) or [GRCh38/hg38](http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz).
 - ```-A```: Gene annotation file. Can instead provide `grch37` or `grch38` to use GENCODE V24 canonical annotation files included with the package. To create custom gene annotation files, use `spliceai/annotations/grch37.txt` in repository as template and provide as full path.

Optional parameters:
 - ```-D```: Maximum distance between the variant and gained/lost splice site (default: 50).
 - ```-M```: Mask scores representing annotated acceptor/donor gain and unannotated acceptor/donor loss (default: 0).
 - ```-B```: Number of predictions to collect before running models on them in batch. (default: 1 (don't batch))
 - ```-T```: Internal Tensorflow `predict()` batch size if you want something different from the `-B` value. (default: the `-B` value)
 - ```-V```: Enable verbose logging during run
 - ```-t```: Specify a location to create the temporary files
 - ```-G```: Specify the GPU(s) to run on : either indexed (eg : 0,2) or 'all'. (default: 'all')
 - ```-S```: Simulate *n* multiple GPUs on a single physical device. Used for development only, currently all values above 2 crashed due to memory issues. (default: 0)
 - ```-P```: Port to use when connecting to the socket (default: 54677, only used in batch mode).

**Batching Considerations:** 

When setting the batching parameters, be mindful of the system and gpu memory of the machine you 
are running the script on. Feel free to experiment, but some reasonable `-T` numbers would be 64/128. CPU memory is larger, and increasing `-B` might further improve performance.

*Batching Performance Benchmarks:*
- Input data: GATK generated WES sample with ~ 90K variants in genome build GRCh37.
- Total predictions made : 174,237
- invitae v2 mainly implements logic to prioritize full batches while predicting 
- settings : 
    - invitae & invitae v2 : B = T = 64
    - invitae v2 optimal : on V100 : B = 4096 ; T = 256 -- on K80/GeForce : B = 4096 ; T = 64

*Benchmark results*

| Type                                 | Implementation        | Total Time | Speed (predictions / hour) |
|--------------------------------------|-----------------------|------------|----------------------------|
| CPU (intel i5-8365U)<sup>a</sup>     | illumina              | ~100h      | ~1000 pred/h               |
|                                      | invitae               | ~39h       | ~4500 pred/h               |
|                                      | invitae v2            | ~35h       | ~5000 pred/h               |
|                                      | invitae v2 optimal    | ~35h       | ~5000 pred/h               | 
| K80 GPU (AWS p2.large)               | illumina</sup>b</sup> | ~25 h      | ~7000 pred/h               |
|                                      | invitae               | 242m       | ~43,000 pred / h           |
|                                      | invitae v2            | 213m       | ~50,000 pred / h           |
|                                      | invitae v2 optimal    | 188 m      | ~56,000 pred / h           |
| GeForce RTX 2070 SUPER GPU (desktop) | illumina</sup>b</sup> | ~10 h      | ~ 17,000 pred/h            |
|                                      | invitae               | 76m        | ~137,000 pred / h          |
|                                      | invitae v2            | 63m        | ~166,000 pred / h          |
|                                      | invitae v2 optimal    | 52m        | ~200,000 pred / h          |
| V100 GPU (AWS p3.xlarge)             | illumina<sup>b</sup>  | ~10h       | ~18,000 pred/h             |
|                                      | invitae               | 78m        | ~135,000 pred / h          |
|                                      | invitae v2            | 54m        | ~190,000 pred / h          |  
|                                      | invitae v2 optimal    | 31 m       | ~335,000 pred / h          |
  

<sup>(a)</sup> : Extrapolated from first 500 variants

<sup>(b)</sup> : Illumina implementation showed a memory leak with the installed versions of tf/keras/.... Values extrapolated from incomplete runs at the point of OOM. 

*Note:* On a p3.8xlarge machine, hosting 4 V100 GPU's, we were able reach 1,379,505 predictions/hour ! This is a nearly linear scale-up.

### Details of SpliceAI INFO field:

| ID     | Description                    |
|--------|--------------------------------|
| ALLELE | Alternate allele               |
| SYMBOL | Gene symbol                    |
| DS_AG  | Delta score (acceptor gain)    |
| DS_AL  | Delta score (acceptor loss)    |
| DS_DG  | Delta score (donor gain)       |
| DS_DL  | Delta score (donor loss)       |
| DP_AG  | Delta position (acceptor gain) |
| DP_AL  | Delta position (acceptor loss) |
| DP_DG  | Delta position (donor gain)    |
| DP_DL  | Delta position (donor loss)    |

Delta score of a variant, defined as the maximum of (DS_AG, DS_AL, DS_DG, DS_DL), ranges from 0 to 1 and can be interpreted as the probability of the variant being splice-altering. In the paper, a detailed characterization is provided for 0.2 (high recall), 0.5 (recommended), and 0.8 (high precision) cutoffs. Delta position conveys information about the location where splicing changes relative to the variant position (positive values are downstream of the variant, negative values are upstream).

### Examples
A sample input file and the corresponding output file can be found at `examples/input.vcf` and `examples/output.vcf` respectively. The output `T|RYR1|0.00|0.00|0.91|0.08|-28|-46|-2|-31` for the variant `19:38958362 C>T` can be interpreted as follows:
* The probability that the position 19:38958360 (=38958362-2) is used as a splice donor increases by 0.91.
* The probability that the position 19:38958331 (=38958362-31) is used as a splice donor decreases by 0.08.

Similarly, the output `CA|TTN|0.07|1.00|0.00|0.00|-7|-1|35|-29` for the variant `2:179415988 C>CA` has the following interpretation:
* The probability that the position 2:179415981 (=179415988-7) is used as a splice acceptor increases by 0.07.
* The probability that the position 2:179415987 (=179415988-1) is used as a splice acceptor decreases by 1.00.

### Frequently asked questions

**1. Why are some variants not scored by SpliceAI?**

SpliceAI only annotates variants within genes defined by the gene annotation file. Additionally, SpliceAI does not annotate variants if they are close to chromosome ends (5kb on either side), deletions of length greater than twice the input parameter ```-D```, or inconsistent with the reference fasta file.

**2. What are the differences between raw (```-M 0```) and masked (```-M 1```) precomputed files?**

The raw files also include splicing changes corresponding to strengthening annotated splice sites and weakening unannotated splice sites, which are typically much less pathogenic than weakening annotated splice sites and strengthening unannotated splice sites. The delta scores of such splicing changes are set to 0 in the masked files. We recommend using raw files for alternative splicing analysis and masked files for variant interpretation.

**3. Can SpliceAI be used to score custom sequences?**

Yes, install SpliceAI and use the following script:  

```python
from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import numpy as np

input_sequence = 'CGATCTGACGTGGGTGTCATCGCATTATCGATATTGCAT'
# Replace this with your custom sequence

context = 10000
paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
models = [load_model(resource_filename('spliceai', x)) for x in paths]
x = one_hot_encode('N'*(context//2) + input_sequence + 'N'*(context//2))[None, :]
y = np.mean([models[m].predict(x) for m in range(5)], axis=0)

acceptor_prob = y[0, :, 1]
donor_prob = y[0, :, 2]
```

### Modifications to Original

**Batching Support** - Invitae (_December 2021_)

* Adds new command line parameters, `--prediction-batch-size` and `--tensorflow-batch-size` to support batching variants to optimize prediction utilization on a GPU 
* Adds a `VCFPredictionBatch` class that manages collection the VCF records, placing them in batches based on the encoded tensor size. Once the batch size is reached, predictions are run in batches, then output is written back in the original order reassembling the annotations for the VCF record. Each VCF record has a lookup key for where each of the ref/alts are within their batches, so it knows where to grab the results during reassembly
* Breaks out code in the existing `get_delta_scores` method into reusable methods used in the batching and the original source code. This way the batching code can utilize the same logic inside that method while still maintaining the original version
* Adds batch utility methods that split up what was all previously done in `get_delta_scores`. `encode_batch_record` handles what was in the first half, taking in the VCF record and generating one-hot encoded matrices for the ref/alts. `extract_delta_scores` handles the second half of the `get_delta_scores` by reassembling the annotations based on the batched predictions
* Adds test cases to run a small file using a generated FASTA reference to test if the results are the same with no batching and with different batching sizes
* Slightly modifies the entrypoint of running the code to allow for easier unit testing. Being able to pass in what would normally come from the argparser

**Multi-GPU support** - Geert Vandeweyer (_November 2022_)

* Offload more code to CPU (eg np to tensor conversion) to *only* perform predictions on the GPU
* Implement queuing system to always have full batches ready for prediction
* Implement new parameter, `--tmpdir` to support a custom tmp folder to store prepped batches
* Implement socket-based client/server approach to scale over multiple GPUs


### Contact
Kishore Jaganathan: kjaganathan@illumina.com

Geert Vandeweyer (This implementation) : geert.vandeweyer@uza.be
