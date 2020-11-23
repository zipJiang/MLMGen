"""We generate list of sentences for each condition,
and combine them together with the yaml configuration.
"""
import yaml
import argparse
from model.mlm import get_bert, ConfigurationMLMSampler, MLMSampler


def get_parser():
    """
    """
    parser = argparse.ArgumentParser(
        """Running MLMGen for a couple of inputs.
        """
    )
    parser.add_argument('--num_samples_per_item', type=int,
                        required=True, action='store', dest='num_samples')
    parser.add_argument('--config_file', action='store', type=str,
                        dest='config_file', required=True)
    parser.add_argument('--device', action='store', dest='device',
                        type=str, required=False, default='cpu')

    return parser


def generate_sentences():
    """Main execution function.
    """
    parser = get_parser()
    args = parser.parse_args()

    # get a model
    model_config = ConfigurationMLMSampler(device=args.device)
    sampler = MLMSampler(model_config, get_bert)

    with open(args.config_file, 'r', encoding='utf-8') as file_:
        configuration = yaml.load(file_)
        for keyword_list in configuration['items']:
            for instance_list in keyword_list.values():
                for instance in instance_list:
                    results, forbidden_ids = sampler.sample_sentences(
                        args.num_samples,
                        word=instance['keyword'],
                        triggers=instance['triggers'])

                # add these back into the instance
                instance['generated'] = results
                instance['forbidden_ids'] = forbidden_ids

    # dump the generation
    with open('outs.yaml', 'w', encoding='utf-8') as file_:
        yaml.dump(configuration, file_)


if __name__ == '__main__':
    """Running the generation process.
    """
    generate_sentences()
