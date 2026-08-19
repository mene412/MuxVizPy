import numpy as np

def writeComponent(fname: str, ensemble: list[np.ndarray]) -> None:
    """
    Writes a list of arrays (with variable length) to a text file, one per line.

    Parameters
    ----------
    fname : str
        Path to the output text file.
    ensemble : list of np.ndarray
        List of integer arrays to be written. Each array is written on a separate line.
    """
    with open(fname, "w") as cio:
        for i in range(len(ensemble)):
            cio.write(" ".join(map(str,ensemble[i]))+"\n")

def readComponent(fname: str) -> list[np.ndarray]:
    """
    Reads a text file written by `writeComponent`, parsing each line into an array of integers.

    Parameters
    ----------
    fname : str
        Path to the input file.

    Returns
    -------
    list of np.ndarray
        List of integer arrays, one per line of the file.
    """
    with open(fname, "r") as fread:
        read_list = fread.readlines()
    var = []
    for i in range(len(read_list)):
        if read_list[i][:-1]=="":
            var.append(np.array([]))
        else:
            var.append(np.array(list(map(int,read_list[i][:-1].split(" ")))))
    return var

def get_names(nodes_list: np.ndarray, net) -> np.ndarray:
    """
    Retrieves node names corresponding to a list of node indices in a VirusMultiplex object.

    Parameters
    ----------
    nodes_list : np.ndarray
        Array of integer node indices.
    net : object
        Instance of VirusMultiplex or similar, containing `node_map`.

    Returns
    -------
    np.ndarray
        Array of node names corresponding to the given indices.
    """
    keys = np.array(list(net.node_map.keys()))
    return keys[nodes_list]