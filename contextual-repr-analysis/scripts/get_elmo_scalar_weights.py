"""
Given a model archive, find an ElmoEmbedder and get the scalar
weighting learned by ELMo.
"""
import argparse
import json
import logging

from allennlp.models import load_archive
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules import Elmo
from allennlp.modules.scalar_mix import ScalarMix
import torch
logger = logging.getLogger(__name__)


def main():
    archive = load_archive(args.model_path)
    model = archive.model

    # Get all the torch modules in the model
    for key, module in model._modules.items():
        # Check TextFieldEmbedders in the modules for ElmoEmbedders
        if isinstance(module, TextFieldEmbedder):
            for embedder_key, embedder_module in module._modules.items():
                # Look for a ElmoTokenEmbedder in the TextFieldEmbedder
                if isinstance(embedder_module, ElmoTokenEmbedder):
                    scalar_weights = get_scalar_weights_from_elmo_token_embedder(
                        embedder_module)
                    # Print these scalar weights
                    print("Found ElmoTokenEmbedder {} in TextFieldEmbedder {} "
                          "with scalar weights: ".format(embedder_key, key))
                    print(json.dumps(scalar_weights, indent=4))
                    print("=" * 79)
        # Check for Elmo modules directly as well (e.g. with the BCN)
        if isinstance(module, Elmo):
            scalar_weights = get_scalar_weights_from_elmo(module)
            # Print these scalar weights
            print("Found Elmo {} with scalar weights: ".format(key))
            print(json.dumps(scalar_weights, indent=4))
            print("=" * 79)
        # Check for ScalarMix
        if isinstance(module, ScalarMix):
            scalars = [scalar.item() for scalar in module.scalar_parameters]
            gamma = module.gamma.item()
            normed_scalars = torch.nn.functional.softmax(torch.cat(
                [parameter for parameter in module.scalar_parameters]), dim=0)
            normed_scalars = torch.split(normed_scalars, split_size_or_sections=1)
            normed_scalars = [normed_scalar.item() for normed_scalar in normed_scalars]
            scalar_weights["scalar_weights"] = {
                "scalars": scalars, "normed_scalars": normed_scalars, "gamma": gamma}
            print("Found ScalarMix {} with scalar weights: ".format(key))
            print(json.dumps(scalar_weights, indent=4))
            print("=" * 79)


def get_scalar_weights_from_elmo_token_embedder(elmo_token_embedder):
    # Get the Elmo module
    elmo_module = elmo_token_embedder._elmo

    # Get the scalar mixes from the Elmo module
    return get_scalar_weights_from_elmo(elmo_module)


def get_scalar_weights_from_elmo(elmo_module):
    scalar_weights = {}
    for idx, scalar_mix in enumerate(elmo_module._scalar_mixes):
        scalars = [scalar.item() for scalar in scalar_mix.scalar_parameters]
        gamma = scalar_mix.gamma.item()
        normed_scalars = torch.nn.functional.softmax(torch.cat(
            [parameter for parameter in scalar_mix.scalar_parameters]), dim=0)
        normed_scalars = torch.split(normed_scalars, split_size_or_sections=1)
        normed_scalars = [normed_scalar.item() for normed_scalar in normed_scalars]

        scalar_weights["scalar_weights_{}".format(idx)] = {
            "scalars": scalars, "normed_scalars": normed_scalars, "gamma": gamma}
    return scalar_weights


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    parser = argparse.ArgumentParser(
        description=("Given a model archive, find an ElmoEmbedder "
                     "and get the scalar weighting learned by ELMo."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", type=str, required=True,
                        help=("Path to the model archive generated by AllenNLP."))
    args = parser.parse_args()
    main()
