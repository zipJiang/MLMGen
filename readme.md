### Masked LM for generation as sampling

This repo is dedicated to training and sampling from a BERT (RoBERTa, ELECTRA-gen) model, where user could specify keywords and semantic trigger pairs.

To test sample generation go to `/model/` and run:

```shellscript
python[3] mlm.py
```

A sample yaml file is presented in `/yamls/` where you could specify arguments for batched generation. To run generation from yaml file, go to `/step/` and run 

```shellscript
python[3] generate_sentence_batches.py --config_file {CONFIG_FILE}\
    --num_samples_per_item {SAMPLE_PER_ITEM}\
    --device {DEVICE}
```

An example is presented in `run_generation.sh`

Notice that to successfully run this you need to add the repository root directory to your `$PYTHONPATH`.
