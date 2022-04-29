"""
Main module for model building:
  - main(...): Main function for running model building, given a model building config dict
    * Loads a SONATA connectome
    * Extracts connectivity specific data
    * Fits a model to data
    * Stores the data and model to disk
    * Visualizes and compares data and model
  - create_model_config_per_pathway(...): Pathway-specific model building wrapper function
"""

import importlib
import json
import os.path
import pickle
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from bluepysnap.circuit import Circuit

from connectome_manipulator import log
from connectome_manipulator.model_building import model_types
from connectome_manipulator.access_functions import get_node_ids, get_edges_population


# NOT USED ANY MORE
# def get_model(model, model_inputs, model_params):
#     """Returns model function from string representation [so any model function can be saved to file]."""
#     input_str = ','.join(model_inputs + ['model_params=model_params']) # String representation of input variables
#     input_param_str = ','.join(model_inputs + list(model_params.keys())) # String representation of input variables and model parameters
#     model_param_str = ','.join(model_inputs + ['**model_params']) # String representation propagating model parameters

#     inner_model_str = f'lambda {input_param_str}: {model}'
#     full_model_str = f'lambda {input_str}: ({inner_model_str})({model_param_str})' # Use nested lambdas to bind local variables

#     model_fct = eval(full_model_str) # Build function

#     # print(f'INFO: Model function: {inner_model_str}')

#     return model_fct


def create_model_config_per_pathway(model_config, grouped_by, src_sel_key='sel_src', dest_sel_key='sel_dest'):
    """Create model config dict for pathways between all pairs of groups (e.g. layer, mtype, ...)."""
    # Check model config
    assert 'model' in model_config.keys(), 'ERROR: "model" key missing in model_config!'
    assert 'working_dir' in model_config.keys(), 'ERROR: "working_dir" key missing in model_config!'
    assert 'out_dir' in model_config.keys(), 'ERROR: "out_dir" key missing in model_config!'
    assert 'name' in model_config['model'].keys(), 'ERROR: "name" key missing in model_config["model"]!'
    assert 'fct' in model_config['model'].keys(), 'ERROR: "fct" key missing in model_config["model"]!'
    assert 'kwargs' in model_config['model']['fct'].keys(), 'ERROR: "kwargs" key missing in model_config["model"]["fct"]!'

    # Load circuit
    circuit_config = model_config['circuit_config']
    circuit = Circuit(circuit_config)
    print(f'INFO: Circuit loaded: {circuit_config}')

    # Select edge population [assuming exactly one edge population in given edges file]
    edges = get_edges_population(circuit)

    # Select corresponding source/target nodes populations
    src_nodes = edges.source
    tgt_nodes = edges.target

    # Find pathways between pairs of groups (within current selection)
    sel_src = model_config['model']['fct']['kwargs'].get(src_sel_key)
    if sel_src is not None:
        assert isinstance(sel_src, dict), 'ERROR: Source node selection must be a dict or empty!' # Otherwise, it cannot be merged with pathway selection
    sel_dest = model_config['model']['fct']['kwargs'].get(dest_sel_key)
    if sel_dest is not None:
        assert isinstance(sel_dest, dict), 'ERROR: Target node selection must be a dict or empty!' # Otherwise, it cannot be merged with pathway selection

    assert grouped_by in src_nodes.property_names, f'ERROR: "{grouped_by}" property not found in source nodes!'
    assert grouped_by in tgt_nodes.property_names, f'ERROR: "{grouped_by}" property not found in target nodes!'

    node_ids_src = get_node_ids(src_nodes, sel_src)
    node_ids_dest = get_node_ids(tgt_nodes, sel_dest)

    src_types = list(np.unique(src_nodes.get(node_ids_src, properties=grouped_by)))
    tgt_types = list(np.unique(tgt_nodes.get(node_ids_dest, properties=grouped_by)))

    # Create list of model configs per pathway
    model_build_name = model_config['model']['name']
    model_config_pathways = []
    for s in src_types:
        for t in tgt_types:
            m_dict = deepcopy(model_config)
            if sel_src is None:
                m_dict['model']['fct']['kwargs'].update({src_sel_key: {grouped_by: s}})
            else:
                m_dict['model']['fct']['kwargs'][src_sel_key].update({grouped_by: s})
            if sel_dest is None:
                m_dict['model']['fct']['kwargs'].update({dest_sel_key: {grouped_by: t}})
            else:
                m_dict['model']['fct']['kwargs'][dest_sel_key].update({grouped_by: t})
            m_dict['model']['name'] += f'__{grouped_by}__{str(s).replace(":", "_")}-{str(t).replace(":", "_")}'
            m_dict['working_dir'] = os.path.join(m_dict['working_dir'], model_build_name + f'__{grouped_by}_pathways')
            m_dict['out_dir'] = os.path.join(m_dict['out_dir'], model_build_name + f'__{grouped_by}_pathways')
            model_config_pathways.append(m_dict)

    print(f'INFO: Created model configurations for {len(model_config_pathways)} pathways between {len(src_types)}x{len(tgt_types)} {grouped_by}{"e" if grouped_by[-1] == "s" else ""}s')

    return model_config_pathways


def main(model_config_input, show_fig=False, force_recomp=False):  # pragma: no cover
    """Main entry point for connectome model building."""
    # Check model building config(s)
    if not isinstance(model_config_input, list):
        assert isinstance(model_config_input, dict), 'ERROR: model_config_input must be of type list or dict!'
        model_config_input = [model_config_input]

    if len(model_config_input) > 1:
        print(f'INFO: Building {len(model_config_input)} models: {model_config_input[0]["model"]["name"]}..{model_config_input[-1]["model"]["name"]}')

    # Run model building
    for midx, model_config in enumerate(model_config_input):
        if len(model_config_input) > 1:
            print(f'\n>>> BUILDING MODEL {midx + 1}/{len(model_config_input)}: {model_config["model"]["name"]} <<<')

        np.random.seed(model_config.get('seed', 123456))

        if np.isscalar(force_recomp):
            force_reextract = force_recomp
            force_rebuild = force_recomp
        else:
            assert len(force_recomp) == 2, 'ERROR: Two "force_recomp" entries expected!'
            force_reextract = force_recomp[0]
            force_rebuild = force_recomp[1]

        # Prepare saving
        model_build_name = model_config['model']['name']
        out_dir = os.path.join(model_config['out_dir'], model_build_name) # Where to put output/figures
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        data_dir = os.path.join(model_config['working_dir'], 'data') # Where to load/put data
        if not os.path.exists(os.path.split(data_dir)[0]):
            os.makedirs(os.path.split(data_dir)[0])

        model_dir = os.path.join(model_config['working_dir'], 'model') # Where to put model
        if not os.path.exists(os.path.split(model_dir)[0]):
            os.makedirs(os.path.split(model_dir)[0])

        # Initialize logger
        log_file = log.logging_init(out_dir, name=__name__.rsplit('.', maxsplit=1)[-1])

        # Prepare computation module
        comp_source = model_config['model']['fct']['source']
        comp_kwargs = model_config['model']['fct']['kwargs']

        comp_module = importlib.import_module('connectome_manipulator.model_building.' + comp_source)
        log.log_assert(hasattr(comp_module, 'extract'), f'ERROR: Model building module "{comp_source}" requires extract() function!')
        log.log_assert(hasattr(comp_module, 'build'), f'ERROR: Model building module "{comp_source}" requires build() function!')
        log.log_assert(hasattr(comp_module, 'plot'), f'ERROR: Model building module "{comp_source}" requires plot() function!')

        # Load circuit
        circuit_config = model_config['circuit_config']
        circuit = Circuit(circuit_config)
        log.info(f'Circuit loaded: {circuit_config}')

        # Extract data (or load from file)
        data_file = os.path.join(data_dir, model_build_name + '.pickle')
        if os.path.exists(data_file) and not force_reextract:
            # Load from file
            log.info(f'Loading data from {data_file}')
            with open(data_file, 'rb') as f:
                data_dict = pickle.load(f)
        else:
            # Compute & save to file
            t_start = time.time()
            data_dict = comp_module.extract(circuit, **comp_kwargs)
            log.info(f'<TIME ELAPSED (data extraction): {time.time() - t_start:.1f}s>')
            log.info(f'Writing data to {data_file}')
            if not os.path.exists(os.path.split(data_file)[0]):
                os.makedirs(os.path.split(data_file)[0])
            with open(data_file, 'wb') as f:
                pickle.dump(data_dict, f)

        # Build model (or load from file)
        model_file = os.path.join(model_dir, model_build_name + '.json')
        if os.path.exists(model_file) and not force_rebuild:
            # Load from file
            log.info(f'Loading model from {model_file}')
            model = model_types.AbstractModel.model_from_file(model_file)
        else:
            # Compute & save to file
            t_start = time.time()
            model = comp_module.build(**data_dict, **comp_kwargs)
            log.info(f'<TIME ELAPSED (model building): {time.time() - t_start:.1f}s>')
            log.info(f'Writing model to {model_file}')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model.save_model(model_dir, model_build_name)

            # Save model config dict for reproducibility
            with open(os.path.join(out_dir, 'model_config.json'), 'w') as f:
                json.dump(model_config, f, indent=2)

        # Visualize data vs. model
        comp_module.plot(**data_dict, **comp_kwargs, model=model, out_dir=out_dir)

        if show_fig:
            plt.show()
        else:
            plt.close("all")
