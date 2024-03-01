"""
Copyright (C) 2018 by Tudor Gheorghiu

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software and associated
documentation files (the "Software"),
to deal in the Software without restriction,
including without limitation the rights to
use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software.
"""
from graphviz import Digraph
import keras


def check_type_and_return_identifier(layer):
    """
        FOR READABILITY
    """
    hidden_layer_handler = 1
    if type(layer) == keras.layers.convolutional.Conv2D or type(layer) == keras.layers.convolutional.DepthwiseConv2D:
        layer_type = "Conv2D"
    elif type(layer) == keras.layers.pooling.MaxPooling2D:
        layer_type = "MaxPooling2D"
    elif type(layer) == keras.layers.core.Dropout:
        layer_type = "Dropout"
    elif type(layer) == keras.layers.core.Flatten:
        layer_type = "Flatten"
    elif type(layer) == keras.layers.core.Activation:
        layer_type = "Activation"
    elif type(layer) == keras.layers.normalization.BatchNormalization:
        layer_type = "BatchNormalization"
    elif type(layer) == keras.engine.input_layer.InputLayer:
        layer_type = None
        hidden_layer_handler = None
    elif type(layer) == keras.layers.merge.Multiply or type(layer) == keras.layers.merge.Add or\
            type(layer) == keras.layers.core.Reshape:
        layer_type = "Arithmetic"
    elif type(layer) == keras.layers.core.Dense:
        layer_type = "Dense"
        hidden_layer_handler = int(str(layer.output_shape).split(",")[1][1:-1])
    else:
        layer_type = "Custom"
    return hidden_layer_handler, layer_type

class KerasArchitectureVisualizer(object):
    """
    Class to visualize keras architecture

    Arguments
    ---------
    filename: str
                where to save the vizualization. (a .gv file)

    title: str
        A title for the graph
    """
    def __init__(self, filename="network.gv", title="My Neural Network"):
        self.filename = filename
        self.title = title
        self.input_layer = 0
        self.hidden_layers_nr = 0
        self.layer_types = []
        self.hidden_layers = []
        self.hidden_layer_layer_num_ref = []
        self.output_layer = 0

    def _reset_states(self):
        self.input_layer = 0
        self.hidden_layers_nr = 0
        self.layer_types = []
        self.hidden_layers = []
        self.output_layer = 0

    def get_image_input_vis(self, the_label, pxls, c, n):
        c.attr(color="white", label=the_label)
        c.node_attr.update(shape="square")
        clr = int(pxls[3][1:-1])
        if (clr == 1):
            clrmap = "Grayscale"
            the_color = "black:white"
        elif (clr == 3):
            clrmap = "RGB"
            the_color = "#e74c3c:#3498db"
        else:
            clrmap = ""
        c.node_attr.update(fontcolor="white", fillcolor=the_color, style="filled")
        n += 1
        c.node(str(n), label="Image\n" + pxls[1] + " x" + pxls[2] + " pixels\n" + clrmap,
               fontcolor="white")
        return n, c

    def visualize(self, model, view=True):
        """Visualizes a keras model.

        Arguments
        ---------
            model: A Keras model instance.

            view: bool
                whether to display the model after generation.
        """
        for l_num, layer in enumerate(model.layers):
            if l_num == 0:
                # First layer
                if type(layer) != keras.engine.input_layer.InputLayer:
                    self.input_layer = int(str(layer.input_shape).split(",")[1][1:-1])
                    self.hidden_layers_nr += 1
                    self.hidden_layer_layer_num_ref.append(l_num)
                hidden_layer, layer_type = check_type_and_return_identifier(layer)
            elif l_num == len(model.layers)-1:
                # Last layer
                self.output_layer = int(str(layer.output_shape).split(",")[1][1:-1])
                hidden_layer, layer_type = None, None
            else:
                # Middle Layers
                self.hidden_layers_nr += 1
                self.hidden_layer_layer_num_ref.append(l_num)
                hidden_layer, layer_type = check_type_and_return_identifier(layer)

            if hidden_layer is not None:
                self.hidden_layers.append(hidden_layer)
            if layer_type is not None:
                self.layer_types.append(layer_type)

            last_layer_nodes = self.input_layer
            nodes_up = self.input_layer
            if(type(model.layers[0]) != keras.layers.core.Dense):
                last_layer_nodes = 1
                nodes_up = 1
                self.input_layer = 1

            n = 0
            g = Digraph('g', filename=self.filename)
            g.graph_attr.update(splines="false", nodesep='1', ranksep='2')
            # Input Layer
            with g.subgraph(name='cluster_input') as c:
                the_label = self.title + '\n\n\n\nInput Layer'
                if (type(model.layers[0]) == keras.layers.core.Dense):
                    if (int(str(model.layers[0].input_shape).split(",")[1][1:-1]) > 10):
                        the_label += " (+" + str(
                            int(str(model.layers[0].input_shape).split(",")[1][1:-1]) - 10) + ")"
                        self.input_layer = 10
                    c.attr(color='white')
                    for i in range(0, self.input_layer):
                        n += 1
                        c.node(str(n))
                        c.attr(label=the_label)
                        c.attr(rank='same')
                        c.node_attr.update(color="#2ecc71", style="filled", fontcolor="#2ecc71",
                                           shape="circle")

                elif (type(model.layers[0]) == keras.layers.convolutional.Conv2D):
                    # Conv2D Input visualizing
                    pxls = str(model.layers[0].input_shape).split(',')
                    n, c = self.get_image_input_vis(the_label, pxls, c, n)

                elif (type(model.layers[0]) == keras.engine.input_layer.InputLayer):
                    pxls = str(model.layers[0].input_shape).split(',')
                    if (len(pxls) == 4):
                        n, c = self.get_image_input_vis(the_label, pxls, c, n)
                    else:
                        raise ValueError("Only have support for image based functional api")
                else:
                    raise ValueError(
                        "Keras Architecture Visualizer: Layer not supported for visualizing")
            for i in range(0, self.hidden_layers_nr):
                layer_copy = model.layers[self.hidden_layer_layer_num_ref[i]]
                with g.subgraph(name="cluster_" + str(i + 1)) as c:
                    if (self.layer_types[i] == "Dense"):
                        c.attr(color='white')
                        c.attr(rank='same')
                        # If hidden_layers[i] > 10, dont include all
                        the_label = ""
                        if (int(str(layer_copy.output_shape).split(",")[1][1:-1]) > 10):
                            the_label += " (+" + str(int(
                                str(layer_copy.output_shape).split(",")[1][1:-1]) - 10) + ")"
                            self.hidden_layers[i] = 10
                        c.attr(labeljust="right", labelloc="b", label=the_label)
                        for j in range(0, self.hidden_layers[i]):
                            n += 1
                            c.node(str(n), shape="circle", style="filled", color="#3498db",
                                   fontcolor="#3498db")
                            for h in range(nodes_up - last_layer_nodes + 1, nodes_up + 1):
                                g.edge(str(h), str(n))
                        last_layer_nodes = self.hidden_layers[i]
                        nodes_up += self.hidden_layers[i]
                    elif (self.layer_types[i] == "Conv2D"):
                        c.attr(style='filled', color='#5faad0')
                        n += 1
                        kernel_size = \
                        str(layer_copy.get_config()['kernel_size']).split(',')[0][1] + "x" + \
                        str(layer_copy.get_config()['kernel_size']).split(',')[1][1: -1]
                        layer_config = layer_copy.get_config()
                        if 'filters' in layer_config:
                            # normal convolution
                            filters = str(model.layers[self.hidden_layer_layer_num_ref[i]].get_config()['filters'])
                        else:
                            # depthwise convolution take channel size * multiplier
                            try:
                                filters = layer.input_shape[3] * layer_config['depth_multiplier']
                            except:
                                print(filters)
                        c.node("conv_.{}".format(n),
                               label="Convolutional Layer\nKernel Size: {}\nFilters: {}".format(kernel_size, filters),
                               shape="square")
                        c.node(str(n), label="{}\nFeature Maps".format(filters), shape="square")
                        g.edge("conv_" + str(n), str(n))
                        for h in range(nodes_up - last_layer_nodes + 1, nodes_up + 1):
                            g.edge(str(h), "conv_" + str(n))
                        last_layer_nodes = 1
                        nodes_up += 1
                    elif (self.layer_types[i] == "MaxPooling2D"):
                        c.attr(color="white")
                        n += 1
                        pool_size = str(model.layers[self.hidden_layer_layer_num_ref[i]].get_config()['pool_size']).split(',')[0][
                                        1] + "x" + \
                                    str(model.layers[self.hidden_layer_layer_num_ref[i]].get_config()['pool_size']).split(',')[1][
                                    1: -1]
                        c.node(str(n), label="Max Pooling\nPool Size: " + pool_size, style="filled",
                               fillcolor="#8e44ad", fontcolor="white")
                        for h in range(nodes_up - last_layer_nodes + 1, nodes_up + 1):
                            g.edge(str(h), str(n))
                        last_layer_nodes = 1
                        nodes_up += 1
                    elif (self.layer_types[i] == "Flatten"):
                        n += 1
                        c.attr(color="white")
                        c.node(str(n), label="Flattening", shape="invtriangle", style="filled",
                               fillcolor="#2c3e50", fontcolor="white")
                        for h in range(nodes_up - last_layer_nodes + 1, nodes_up + 1):
                            g.edge(str(h), str(n))
                        last_layer_nodes = 1
                        nodes_up += 1
                    elif (self.layer_types[i] == "Dropout"):
                        n += 1
                        c.attr(color="white")
                        c.node(str(n), label="Dropout Layer", style="filled", fontcolor="white",
                               fillcolor="#f39c12")
                        for h in range(nodes_up - last_layer_nodes + 1, nodes_up + 1):
                            g.edge(str(h), str(n))
                        last_layer_nodes = 1
                        nodes_up += 1
                    elif (self.layer_types[i] == "Activation"):
                        n += 1
                        c.attr(color="white")
                        fnc = model.layers[self.hidden_layer_layer_num_ref[i]].get_config()['activation']
                        c.node(str(n), shape="octagon", label="Activation Layer\nFunction: " + fnc,
                               style="filled", fontcolor="white", fillcolor="#00b894")
                        for h in range(nodes_up - last_layer_nodes + 1, nodes_up + 1):
                            g.edge(str(h), str(n))
                        last_layer_nodes = 1
                        nodes_up += 1

            with g.subgraph(name='cluster_output') as c:
                if (type(model.layers[-1]) == keras.layers.core.Dense):
                    c.attr(color='white')
                    c.attr(rank='same')
                    c.attr(labeljust="1")
                    for i in range(1, self.output_layer + 1):
                        n += 1
                        c.node(str(n), shape="circle", style="filled", color="#e74c3c",
                               fontcolor="#e74c3c")
                        for h in range(nodes_up - last_layer_nodes + 1, nodes_up + 1):
                            g.edge(str(h), str(n))
                    c.attr(label='Output Layer', labelloc="bottom")
                    c.node_attr.update(color="#2ecc71", style="filled", fontcolor="#2ecc71",
                                       shape="circle")

            g.attr(arrowShape="none")
            g.edge_attr.update(arrowhead="none", color="#707070")
        if view == True:
            g.view()