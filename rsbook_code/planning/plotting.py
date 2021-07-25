def mpl_plot_graph(ax,G,vertex_options={},edge_options={},dims=[0,1],directed=False):
    """Plots a graph G=(V,E) using matplotlib.

    ax is a matplotlib Axes object.

    If states have more than 2 dimensions, you can control the x-y axes
    using the dims argument.
    """
    import numpy as np
    V,E = G
    if len(V)==0:
        return
    X = [v[dims[0]] for v in V]
    Y = [v[dims[1]] for v in V]
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    lines = []
    for e in E:
        x1,y1 = X[e[0]],Y[e[0]]
        x2,y2 = X[e[1]],Y[e[1]]
        lines.append(np.array([[x1,y1],[x2,y2]],dtype=float))
    #convert normal edge options to collection options
    collection_options = {}
    for k,opt in edge_options.items():
        if not k.endswith('s') and k not in ['alpha']:
            collection_options[k+'s'] = np.asarray([opt]*len(lines))
    linecoll = LineCollection(lines,zorder=2,**collection_options)
    ax.add_collection(linecoll)

    ax.scatter(X,Y,zorder=3,**vertex_options)
    
