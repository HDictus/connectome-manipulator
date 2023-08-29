"""Manipulation name: conn_wiring

Description: Special case of connectome rewiring, which wires an empty connectome from scratch, or simply
adds connections to an existing connectome (edges table)!
Only specific properties like source/target node, afferent synapse positions, synapse type
(INH: 0, EXC: 100), and delay (optional) will be generated.
"""

from datetime import datetime, timedelta

import libsonata
import neurom as nm
import numpy as np
import tqdm

from connectome_manipulator import log, profiler
from connectome_manipulator.access_functions import (
    get_attribute,
    get_node_ids,
    get_enumeration,
)
from connectome_manipulator.connectome_manipulation.converters import EdgeWriter
from connectome_manipulator.connectome_manipulation.manipulation import (
    MorphologyCachingManipulation,
)
from connectome_manipulator.model_building import model_types, conn_prob

# IDEAs for improvements:
#   Add model for synapse placement


class ConnectomeWiring(MorphologyCachingManipulation):
    """Special case of connectome rewiring

    Wires an empty connectome from scratch, or simply adds connections to an existing connectome (edges table)!
    Only specific properties like source/target node, afferent synapse positions, synapse type
    (INH: 0, EXC: 100), and delay (optional) will be generated.
    """

    SYNAPSE_PROPERTIES = [
        "@target_node",
        "@source_node",
        "afferent_section_id",
        "afferent_section_pos",
        "afferent_section_type",
        "afferent_center_x",
        "afferent_center_y",
        "afferent_center_z",
        "syn_type_id",
        "delay",
    ]
    PROPERTY_TYPES = {
        "@target_node": "int64",
        "@source_node": "int64",
        "afferent_section_id": "int32",
        "afferent_section_pos": "float32",
        "afferent_section_type": "int16",
        "afferent_center_x": "float32",
        "afferent_center_y": "float32",
        "afferent_center_z": "float32",
        "syn_type_id": "int16",
        "delay": "float32",
    }

    @profiler.profileit(name="conn_wiring")
    def apply(
        self,
        split_ids,
        sel_src=None,
        sel_dest=None,
        pos_map_file=None,
        amount_pct=100.0,
        morph_ext="swc",
        prob_model_spec=None,
        nsynconn_model_spec=None,
        delay_model_spec=None,
        pathway_specs=None,
        **kwargs,
    ):
        """Wiring (generation) of structural connections between pairs of neurons based on given conn. prob. model.

        => Only structural synapse properties will be set: PRE/POST neuron IDs, synapse positions, type, axonal delays
        => Model specs: A dict with model type/attributes or a dict with "file" key pointing to a model file can be passed
        """
        # pylint: disable=arguments-differ
        assert len(kwargs) == 0
        if not prob_model_spec:
            prob_model_spec = {
                "model": "ConnProb1stOrderModel",
            }  # Default 1st-oder model
        if not nsynconn_model_spec:
            nsynconn_model_spec = {
                "model": "NSynConnModel",
            }  # Default #syn/conn model
        if not delay_model_spec:
            delay_model_spec = {
                "model": "LinDelayModel",
            }  # Default linear delay model
        for spec in (prob_model_spec, nsynconn_model_spec, delay_model_spec):
            # AbstractModel insists that "file" is the only key if present
            if "file" not in spec:
                spec["src_type_map"] = self.src_type_map
                spec["tgt_type_map"] = self.tgt_type_map
                spec["pathway_specs"] = pathway_specs
        # pylint: disable=arguments-differ, arguments-renamed
        log.log_assert(0.0 <= amount_pct <= 100.0, "amount_pct out of range!")

        with profiler.profileit(name="conn_wiring/setup"):
            # Intersect target nodes with split IDs and return if intersection is empty
            tgt_node_ids = get_node_ids(self.nodes[1], sel_dest, split_ids)
            num_tgt_total = len(tgt_node_ids)
            if num_tgt_total == 0:  # Nothing to wire
                log.info("No target nodes selected, nothing to wire")
                return
            if amount_pct < 100:
                num_tgt = np.round(amount_pct * num_tgt_total / 100).astype(int)
                tgt_sel = np.random.permutation(
                    np.concatenate(
                        (np.full(num_tgt, True), np.full(num_tgt_total - num_tgt, False)), axis=None
                    )
                )
            else:
                num_tgt = num_tgt_total
                tgt_sel = np.full(num_tgt_total, True)
            if num_tgt == 0:  # Nothing to wire
                log.info("No target nodes selected, nothing to wire")
                return
            # Load connection probability model
            p_model = model_types.AbstractModel.init_model(prob_model_spec)
            log.debug(f'Loaded conn. prob. model of type "{p_model.__class__.__name__}"')

            # Load #synapses/connection model
            nsynconn_model = model_types.AbstractModel.init_model(nsynconn_model_spec)
            log.debug(
                f'Loaded #synapses/connection model of type "{nsynconn_model.__class__.__name__}"'
            )

            # Load delay model (optional)
            if delay_model_spec is not None:
                delay_model = model_types.AbstractModel.init_model(delay_model_spec)
                log.debug(f'Loaded delay model of type "{delay_model.__class__.__name__}"')
            else:
                delay_model = None
                log.debug("No delay model provided")

            # Load position mapping model (optional) => [NOTE: SRC AND TGT NODES MUST BE INCLUDED WITHIN SAME POSITION MAPPING MODEL]
            _, pos_acc = conn_prob.load_pos_mapping_model(pos_map_file)
            if pos_acc is None:
                log.debug("No position mapping model provided")

            # Determine source/target nodes for wiring
            src_node_ids = get_node_ids(self.nodes[0], sel_src)
            src_class = get_attribute(self.nodes[0], "synapse_class", src_node_ids)
            src_mtypes = get_enumeration(self.nodes[0], "mtype", src_node_ids)
            log.log_assert(len(src_node_ids) > 0, "No source nodes selected!")

            tgt_node_ids = tgt_node_ids[tgt_sel]  # Select subset of neurons (keeping order)
            tgt_mtypes = get_enumeration(self.nodes[1], "mtype", tgt_node_ids)

            _src_pop = self.nodes[0]._population  # pylint: disable=protected-access
            _src_sel = libsonata.Selection(src_node_ids)
            raw_src_pos = np.column_stack(
                (
                    _src_pop.get_attribute("x", _src_sel),
                    _src_pop.get_attribute("y", _src_sel),
                    _src_pop.get_attribute("z", _src_sel),
                )
            )  # Raw src positions required for delay computations

            if pos_acc:  # Position mapping provided
                # FIXME: this is going to be VERY SLOW!
                # Get neuron positions (incl. position mapping)
                src_pos = conn_prob.get_neuron_positions(pos_acc, [src_node_ids])[0]
                tgt_pos = conn_prob.get_neuron_positions(pos_acc, [tgt_node_ids])[0]
            else:
                src_pos = raw_src_pos
                _tgt_pop = self.nodes[1]._population  # pylint: disable=protected-access
                _tgt_sel = libsonata.Selection(tgt_node_ids)
                tgt_pos = np.column_stack(
                    (
                        _tgt_pop.get_attribute("x", _tgt_sel),
                        _tgt_pop.get_attribute("y", _tgt_sel),
                        _tgt_pop.get_attribute("z", _tgt_sel),
                    )
                )

            log.info(
                f"Generating afferent connections to {num_tgt} ({amount_pct}%) of {len(tgt_sel)} target neurons in current split (total={num_tgt_total}, sel_src={sel_src}, sel_dest={sel_dest})"
            )

        # Run connection wiring
        self._connectome_wiring_wrapper(
            src_node_ids,
            src_pos,
            src_mtypes,
            src_class,
            morph_ext,
            tgt_node_ids,
            tgt_pos,
            tgt_mtypes,
            p_model,
            nsynconn_model,
            delay_model,
            raw_src_pos,
        )

    @profiler.profileit(name="conn_wiring/wiring")
    def _connectome_wiring_wrapper(
        self,
        src_node_ids,
        src_positions,
        src_mtypes,
        src_class,
        morph_ext,
        tgt_node_ids,
        tgt_positions,
        tgt_mtypes,
        p_model,
        nsynconn_model,
        delay_model,
        raw_src_positions,  # src positions w/o pos mapping (for delays!)
    ):
        """Stand-alone wrapper for connectome wiring."""
        # get morphologies for this selection
        tgt_morphs = self._get_tgt_morphs(morph_ext, libsonata.Selection(tgt_node_ids))

        log_time = datetime.now()
        for tidx, (tgt, morph) in enumerate(zip(tgt_node_ids, tgt_morphs)):
            new_time = datetime.now()
            if (new_time - log_time) / timedelta(minutes=1) > 1:
                log.info(
                    "Processing target node %d out of %d",
                    tidx,
                    len(tgt_node_ids),
                )
                log_time = new_time

            # Determine conn. prob. of all source nodes to be connected with target node (mtype-specific)
            tgt_pos = tgt_positions[
                tidx : tidx + 1, :
            ]  # Get neuron positions (incl. position mapping, if provided)
            p_src = p_model.apply(
                src_pos=src_positions,
                tgt_pos=tgt_pos,
                src_type=src_mtypes,
                tgt_type=[tgt_mtypes[tidx]],
            ).flatten()
            p_src[np.isnan(p_src)] = 0.0  # Exclude invalid values
            # Exclude autapses [ASSUMING node IDs are unique across src/tgt
            # node populations!]
            p_src[src_node_ids == tgt] = 0.0

            # Sample new presynaptic neurons from list of source nodes according to conn. prob.
            src_new_sel = np.random.rand(len(src_node_ids)) < p_src
            src_new = src_node_ids[src_new_sel]  # New source node IDs per connection
            num_new = len(src_new)
            if num_new == 0:
                continue  # Nothing to wire

            # Sample number of synapses per connection (mtype-specific)
            num_syn_per_conn = nsynconn_model.apply(
                src_type=src_mtypes[src_new_sel], tgt_type=tgt_mtypes[tidx]
            )
            syn_conn_idx = np.concatenate(
                [[i] * n for i, n in enumerate(num_syn_per_conn)]
            )  # Create mapping from synapses to connections
            num_gen_syn = len(syn_conn_idx)  # Number of synapses to generate

            # Place synapses randomly on soma/dendrite sections
            # [TODO: Add model for synapse placement??]
            sec_ind = np.hstack(
                [
                    [-1],
                    np.flatnonzero(
                        np.isin(morph.section_types, [nm.BASAL_DENDRITE, nm.APICAL_DENDRITE])
                    ),
                ]
            )

            # Randomly choose section indices
            sec_sel = np.random.choice(sec_ind, len(syn_conn_idx))

            # Randomly choose fractional offset within each section
            off_sel = np.random.rand(len(syn_conn_idx))
            off_sel[sec_sel == -1] = 0.0  # Soma offsets must be zero

            # Type 0: Soma (1: Axon, 2: Basal, 3: Apical)
            type_sel = np.full_like(sec_sel, 0)
            # Synapse positions, computed from section & offset
            pos_sel = np.tile(morph.soma.center.astype(float), (len(sec_sel), 1))
            for idx in np.flatnonzero(sec_sel >= 0):
                type_sel[idx] = morph.section(sec_sel[idx]).type
                pos_sel[idx] = nm.morphmath.path_fraction_point(
                    morph.section(sec_sel[idx]).points, off_sel[idx]
                )
            # syn_type = np.select([src_class[new_edges['@source_node']].to_numpy() == 'INH', src_class[new_edges['@source_node']].to_numpy() == 'EXC'], [np.full(num_gen_syn, 0), np.full(num_gen_syn, 100)]) # INH: 0-99 (Using 0); EXC: >=100 (Using 100)
            syn_type = np.select(
                [
                    src_class[src_new_sel][syn_conn_idx] == "INH",
                    src_class[src_new_sel][syn_conn_idx] == "EXC",
                ],
                [np.full(num_gen_syn, 0), np.full(num_gen_syn, 100)],
            )  # INH: 0-99 (Using 0); EXC: >=100 (Using 100)

            # Assign distance-dependent delays (mtype-specific), based on (generative) delay model (optional)
            # IMPORTANT: Distances for delays are computed in them original coordinate system w/o coordinate transformation!
            kwargs = {}
            if delay_model is not None:
                src_new_pos = raw_src_positions[src_new_sel, :]
                syn_dist = np.sqrt(
                    np.sum((pos_sel - src_new_pos[syn_conn_idx, :]) ** 2, 1)
                )  # Distance from source neurons (soma) to synapse positions on target neuron
                delay = delay_model.apply(
                    distance=syn_dist,
                    src_type=src_mtypes[src_new_sel][syn_conn_idx],
                    tgt_type=tgt_mtypes[tidx],
                )
                if np.isscalar(delay):
                    kwargs["delay"] = np.full(syn_type.shape, delay)
                else:
                    kwargs["delay"] = delay

            # IMPORTANT: Section IDs in NeuroM morphology don't include soma, so they need to be shifted by 1 (Soma ID is 0 in edges table)
            self.writer.append(
                source_node_id=src_new[syn_conn_idx],
                target_node_id=np.full_like(syn_type, tgt),
                afferent_section_id=sec_sel + 1,
                afferent_section_pos=off_sel,
                afferent_section_type=type_sel,
                afferent_center_x=pos_sel[:, 0],
                afferent_center_y=pos_sel[:, 1],
                afferent_center_z=pos_sel[:, 2],
                syn_type_id=syn_type,
                edge_type_id=np.zeros_like(syn_type),
                **kwargs,
            )

    @classmethod
    def connectome_wiring_per_pathway(cls, nodes, pathway_models, seed=0, morph_ext="h5"):
        """Stand-alone connectome wiring per pathway, i.e., wiring pathways using pathway-specific probability/nsynconn/delay models."""
        # Init random seed for connectome building and sampling from parameter distributions
        np.random.seed(seed)

        with_delay = any(d["delay_model"] for d in pathway_models)

        writer = EdgeWriter(None, with_delay=with_delay)
        conn_wiring = cls(nodes, writer)
        src_nodes, tgt_nodes = nodes

        # Loop over pathways
        for pathway_dict in tqdm.tqdm(pathway_models):
            # [OPTIMIZATION: Run wiring of pathways in parallel]

            pre_type = pathway_dict["pre"]
            post_type = pathway_dict["post"]
            prob_model = pathway_dict["prob_model"]
            nsynconn_model = pathway_dict["nsynconn_model"]
            delay_model = pathway_dict["delay_model"]

            # Select source/target nodes
            src_node_ids = src_nodes.ids({"mtype": pre_type})
            src_class = get_attribute(src_nodes, "synapse_class", src_node_ids)
            src_mtypes = get_enumeration(src_nodes, "mtype", src_node_ids)
            src_positions = src_nodes.positions(
                src_node_ids
            ).to_numpy()  # OPTIONAL: Coordinate system transformation may be added here

            tgt_node_ids = tgt_nodes.ids({"mtype": post_type})
            tgt_mtypes = get_enumeration(tgt_nodes, "mtype", tgt_node_ids)
            tgt_positions = tgt_nodes.positions(
                tgt_node_ids
            ).to_numpy()  # OPTIONAL: Coordinate system transformation may be added here

            # Create edges per pathway
            # pylint: disable=protected-access
            conn_wiring._connectome_wiring_wrapper(
                src_node_ids,
                src_positions,
                src_mtypes,
                src_class,
                morph_ext,
                tgt_node_ids,
                tgt_positions,
                tgt_mtypes,
                prob_model,
                nsynconn_model,
                delay_model,
                src_positions,
            )

            # ALTERNATIVE: Write to .parquet file and merge/convert to SONATA later
            # ... connectome_manipulation.edges_to_parquet(edges_table, output_file)
            # ... connectome_manipulation.parquet_to_sonata(input_file_list, output_file, nodes, nodes_files, keep_parquet=False)

        # Merge edges, re-sort, and assign new index
        edges_table = writer.to_pandas()
        edges_table.sort_values(["@target_node", "@source_node"], inplace=True)
        edges_table.reset_index(inplace=True, drop=True)

        return edges_table
