from typing import Dict, List, Tuple, Optional
import copy
import functools

def subtract_lists_(l1: List, l2: List):
    return [x for x in l1 if x not in l2]

def lists_equal_(l1: List, l2: List):
    """
    
    NOTE: this only works if the lists elements are also unique

    """
    return set(l1) == set(l2)

class DirectedNode:
    def __init__(self, id: int, name, uses: List, defines: List):
        self.id = id
        self.name = name
        self.incoming_edges: List[DirectedNode] = []
        self.outgoing_edges: List[DirectedNode] = []

        self.uses = uses
        self.defines = defines

        self.live_in = []
        self.live_out = []

        # Currently only used for DFS search
        self.extra_properties = {"visited": False, "depth": -1}

    def __repr__(self):
        return self.name
    
    def __hash__(self) -> int:
        return hash(self.id)


class DirectedEdge:
    def __init__(self, src: DirectedNode, dst: DirectedNode):
        self.src = src
        self.dst = dst

        self.live_temporaries = []

    def add_live_temporary(self, temporary):
        self.live_temporaries.append(temporary)

    def __repr__(self):
        return f"{self.src} -> {self.dst}"


class DirectedGraph:
    def __init__(self):
        self.nodes: Dict[int, DirectedNode] = {}
        self.edges: List[DirectedEdge] = []
        self.temporaries: List = []

        self.start_node: Optional[DirectedNode] = None

    def add_temporary(self, temporary):
        self.temporaries.append(temporary)

    def add_node(
        self, name, node_id: Optional[int] = None, uses: List = [], defines: List = [], start_node: bool = False
    ) -> DirectedNode:
        for temporary in uses + defines:
            if temporary not in self.temporaries:
                raise Exception(
                    f"Temporary {temporary} not found in temporaries list. Please add it before using it in a node."
                )

        node_id = node_id if node_id else len(self.nodes)
        self.nodes[node_id] = DirectedNode(node_id, name, uses, defines)

        if start_node:
            self.start_node = self.nodes[node_id]

        return self.nodes[node_id]

    def add_edge(self, src_name, dst_name) -> DirectedEdge:
        src = self.get_node_by_name(src_name)
        dst = self.get_node_by_name(dst_name)
        self.edges.append(DirectedEdge(src, dst))
        src.outgoing_edges.append(dst)
        dst.incoming_edges.append(src)

        return self.edges[-1]

    def get_node(self, node_id: int) -> DirectedNode:
        return self.nodes[node_id]

    def get_node_by_name(self, name: str) -> DirectedNode:
        for node in self.nodes.values():
            if node.name == name:
                return node
        raise Exception(f"Node with name {name} not found.")

    def dfs(self, name):
        node = self.get_node_by_name(name)
        node.extra_properties["depth"] = 0
        self._dfs(node)

        return sorted(self.nodes.values(), key=lambda node: node.extra_properties["depth"])
    
    def _dfs(self, node):
        node.extra_properties["visited"] = True
        for edge in node.outgoing_edges:
            if not edge.extra_properties["visited"]:
                edge.extra_properties["depth"] = node.extra_properties["depth"] + 1
                self._dfs(edge)

    def set_start_node(self, name):
        self.start_node = self.get_node_by_name(name)

    def calculate_liveness(self):
        if not self.start_node:
            raise Exception("No start node found. Please specify a start node before calculating liveness.")

        for node in self.nodes.values():
            node.live_in = []
            node.live_out = []

        node_order = self.dfs(self.start_node.name)
        node_order.reverse()

        print(node_order)

        changed = True
        while changed:
            for node in node_order:
                live_in = node.uses + subtract_lists_(node.live_in, node.defines)
                #live_out = [s.live_in for s in node.outgoing_edges]
                live_out = functools.reduce(lambda a, b: a + b.live_in, node.outgoing_edges, [])

                print(live_out, live_in)

                if lists_equal_(live_in, node.live_in) and lists_equal_(live_out, node.live_out):
                    changed = False

                node.live_in = live_in
                node.live_out = live_out
            print()
        
        for edge in self.edges:
            edge.live_temporaries = list(set(edge.src.live_out) | set(edge.dst.live_in))

        for node in self.nodes.values():
            print(node.name, node.live_out, node.live_in)

    def __repr__(self):
        edges = []

        for edge in self.edges:
            edge_str = f'"{edge.src.name}" -> "{edge.dst.name}"'
            if edge.live_temporaries:
                edge_str += f' [label="{" ".join(edge.live_temporaries)}"]'
            edge_str += '; '
            edges.append(edge_str)

        edges_str = " ".join(edges)
        # nodes_str = " ".join(['"' + node.name +'"' for node in self.nodes.values()])

        # return f"digraph {{rankdir=LR; {edges_str}{{rank=same; {nodes_str}}}}}"
        return f"digraph {{{edges_str}}}"


class Node:
    def __init__(self, id: int, name, color=None):
        self.id = id
        self.name = name
        self.adjacent: List[Node] = []
        self.adjacent_removed: List[Node] = []
        self.move_list: List[Node] = []
        self.move_list_removed: List[Node] = []
        self.color = color

    def __repr__(self):
        if self.color:
            return f"Node({self.name}, {self.color})"
        return f"Node({self.name})"

    def get_move(self):
        """Gets the move list of the node

        Returns:
            List[Node]: The move list of the node
        """
        return self.move_list

    def count_move(self):
        """Gets the number of moves

        Returns:
            int: The number of moves
        """
        return len(self.get_move())

    def get_adjacent(self):
        """Gets the adjacency list of the node

        Returns:
            List[Node]: The adjacency list of the node
        """
        return self.adjacent

    def count_adjacent(self):
        """Gets the number of adjacent nodes

        Returns:
            int: The number of adjacent nodes
        """
        return len(self.get_adjacent())

    def __eq__(self, other) -> bool:
        return self.id == other.id

    def __lt__(self, other) -> bool:
        return self.count_adjacent() < other.count_adjacent()

    def __hash__(self) -> int:
        return hash(self.id)


class CoalescedNode(Node):
    def __init__(self, id: int, nodes: List[Node], color=None):
        super().__init__(id, "-".join([str(node.name) for node in nodes]), color)
        self.nodes = nodes

    def __repr__(self):
        if self.color:
            return f"CoalescedNode({self.nodes}, {self.color})"
        return f"CoalescedNode({self.nodes})"

    def add_node(self, node):
        """Adds a node to the coalesced node

        Args:
            node (Node): The node to add
        """
        self.nodes.append(node)
        self.name = "-".join([str(node.name) for node in self.nodes])


class Graph:
    def __init__(self) -> None:
        self.nodes: Dict[int, Node] = {}
        self.adjacency_matrix = []

    def __repr__(self):
        edges = []

        for node in self.nodes.values():
            for adjacent_node in node.get_adjacent():
                if not f"{adjacent_node.name} -- {node.name};" in edges:
                    edges.append(f"{node.name} -- {adjacent_node.name};")
            for move_node in node.get_move():
                if not f"{move_node.name} -- {node.name} [style=dotted];" in edges:
                    edges.append(f"{node.name} -- {move_node.name} [style=dotted];")

        return "graph {{{}}}".format(" ".join(edges))

    def add_node(self, name, color=None, id=None) -> Node:
        """Creates and adds a node to the graph

        Arguments:
            name {str} -- The name of the node
            color (optional) {any} -- The color of the node
            id (optional) {int} -- The id of the node

        Returns:
            Node -- The created node
        """
        node_id = id if id else len(self.nodes)
        node = Node(node_id, name, color)
        self.add_existing_node(node)
        return node

    def add_existing_node(self, node) -> None:
        """Adds an existing node to the graph

        Arguments:
            node {Node} -- The node to add
        """
        self.nodes[node.id] = node
        for row in self.adjacency_matrix:
            row.append(0)
        self.adjacency_matrix.append([0] * len(self.nodes))

    def get_node_by_name(self, name) -> Node:
        """Finds the node with the given name

        Arguments:
            name {any} -- The name of the node
        """
        for node in self.nodes.values():
            if node.name == name:
                return node

        raise Exception(f"Node {name} not found")

    def add_interference_edge(self, name1, name2) -> None:
        """Adds an interference edge between two nodes

        Arguments:
            name1 {any} -- The name of the first node
            name2 {any} -- The name of the second node
        """
        node1 = self.get_node_by_name(name1)
        node2 = self.get_node_by_name(name2)

        node1_adjacency_matrix_index = self.get_adjacency_matrix_index(name1)
        node2_adjacency_matrix_index = self.get_adjacency_matrix_index(name2)

        self.adjacency_matrix[node1_adjacency_matrix_index][
            node2_adjacency_matrix_index
        ] = 1
        self.adjacency_matrix[node2_adjacency_matrix_index][
            node1_adjacency_matrix_index
        ] = 1

        node1.adjacent.append(node2)
        node2.adjacent.append(node1)

    def add_move_edge(self, name1, name2) -> None:
        """Adds a move edge between two nodes

        Arguments:
            name1 {any} -- The name of the first node
            name2 {any} -- The name of the second node
        """
        node1 = self.get_node_by_name(name1)
        node2 = self.get_node_by_name(name2)

        node1_adjacency_matrix_index = self.get_adjacency_matrix_index(name1)
        node2_adjacency_matrix_index = self.get_adjacency_matrix_index(name2)

        self.adjacency_matrix[node1_adjacency_matrix_index][
            node2_adjacency_matrix_index
        ] = -1
        self.adjacency_matrix[node2_adjacency_matrix_index][
            node1_adjacency_matrix_index
        ] = -1

        node1.move_list.append(node2)
        node2.move_list.append(node1)

    def check_adjacency(self, name1, name2) -> bool:
        """Checks if two nodes are adjacent

        Arguments:
            name1 {any} -- The name of the first node
            name2 {any} -- The name of the second node

        Returns:
            bool -- True if the nodes are adjacent, False otherwise
        """
        node1 = self.get_node_by_name(name1)
        node2 = self.get_node_by_name(name2)
        return self.adjacency_matrix[node1.id][node2.id] == 1

    def get_adjacent(self, name) -> List[Node]:
        """Gets a list of nodes adjacent to the node with the given name

        Arguments:
            name {any} -- The name of the node

        Returns:
            List[Node] -- The list of adjacent nodes
        """
        return self.get_node_by_name(name).get_adjacent()

    def get_move_list(self, name) -> List[Node]:
        """Gets a list of nodes in the move list of the node with the given name

        Arguments:
            name {any} -- The name of the node

        Returns:
            List[Node] -- The list of nodes in the move list
        """
        return self.get_node_by_name(name).get_move()

    def count_adjacent(self, name) -> int:
        """Gets the number of nodes adjacent to the node with the given name

        Arguments:
            name {any} -- The name of the node

        Returns:
            int -- The number of adjacent nodes
        """
        return self.get_node_by_name(name).count_adjacent()

    def get_adjacency_matrix_index(self, name) -> int:
        """Gets the index of the node with the given name in the adjacency matrix

        Arguments:
            name {any} -- The name of the node

        Returns:
            int -- The index of the node in the adjacency matrix
        """
        return list(self.nodes.keys()).index(self.get_node_by_name(name).id)


class InterferenceGraph(Graph):
    def __init__(self, registers: List) -> None:
        super().__init__()
        self.registers = registers
        self.simplified_nodes: Dict[int, Node] = {}
        self.running_node_id = 0

    def add_node(self, name, register=None) -> Node:
        """Creates and adds a node to the graph

        Arguments:
            name {str} -- The name of the node
            register (optional) {any} -- The register the node is assigned to

        Returns:
            Node -- The created node
        """
        node = super().add_node(name, register, self.running_node_id)
        self.running_node_id += 1
        return node

    def take_snapshot(self):
        """Takes a snapshot of the current state of the graph"""
        self.snapshot = copy.deepcopy(self)  # type: ignore

    def remove_node(self, name) -> Node:
        """Removes a node from the graph

        Arguments:
            name {any} -- The name of the node to remove

        Returns:
            Node -- The removed node
        """
        node = self.get_node_by_name(name)

        for i in range(len(self.adjacency_matrix)):
            del self.adjacency_matrix[i][self.get_adjacency_matrix_index(name)]
        del self.adjacency_matrix[self.get_adjacency_matrix_index(name)]

        for adjacent in node.get_adjacent():
            adjacent.adjacent.remove(node)
            adjacent.adjacent_removed.append(node)
        node.adjacent = node.adjacent
        node.adjacent = []
        for move in node.get_move():
            move.move_list.remove(node)
            move.move_list_removed.append(node)
        node.move_list = node.move_list
        node.move_list = []

        self.nodes.pop(node.id)
        return node

    def simplify(self, num_to_simplify=1) -> List[Node]:
        """Simplifies the graph by removing non-move-related and non-precolored nodes with degree less than self.registers (removes the node with the highest degree first)

        Arguments:
            num_to_simplify {int} -- The number of nodes to simplify (default: {1})

        Returns:
            List[Node] -- The list of nodes that were removed
        """
        simplified_nodes = []
        for _ in range(num_to_simplify):
            for node in sorted(interference_graph.nodes.values(), reverse=True):
                if (
                    node.count_adjacent() < len(self.registers)
                    and node.count_move() == 0
                    and node.color == None
                ):
                    self.remove_node(node.name)
                    self.simplified_nodes[node.id] = node
                    simplified_nodes.append(node)
                    break
        return simplified_nodes

    def can_coalesce(self, node1: Node, node2: Node, method="briggs") -> bool:
        """Checks if two nodes can be coalesced

        Arguments:
            node1 {Node} -- The first node
            node2 {Node} -- The second node
            method {"briggs" | "george" } -- The method to use to check if the nodes can be coalesced (default: {"briggs"})

        Returns:
            bool -- True if the nodes can be coalesced, False otherwise
        """
        if method == "briggs":
            return len(set(node1.get_adjacent() + node2.get_adjacent())) < len(
                self.registers
            )
        elif method == "george":
            can_coalesce = True
            for adjacent in node1.get_adjacent():
                if not (
                    adjacent in node2.get_adjacent()
                    or len(adjacent.get_adjacent()) < len(self.registers)
                ):
                    can_coalesce = False
                    break
            return can_coalesce
        else:
            raise Exception(f"Invalid method: {method}")

    def coalesce(self, num_to_coalesce=1) -> List[CoalescedNode]:
        """Coalesces nodes in the graph (removes the node with the highest degree first)

        Arguments:
            num_to_coalesce {int} -- The number of nodes to coalesce (default: {1})

        Returns:
            List[CoalescedNode] -- The list of nodes that were coalesced
        """
        coalesced_nodes = []
        for _ in range(num_to_coalesce):
            for node in sorted(interference_graph.nodes.values(), reverse=True):
                for move in node.get_move():
                    if self.can_coalesce(node, move, method="george"):
                        coalesced_node = CoalescedNode(
                            self.running_node_id, [node, move]
                        )
                        self.running_node_id += 1

                        self.add_existing_node(coalesced_node)

                        combined_adjacent = list(
                            set(node.get_adjacent() + move.get_adjacent())
                        )
                        combined_move = list(set(node.get_move() + move.get_move()))
                        combined_move = [
                            node for node in combined_move if node not in [move, node]
                        ]

                        self.remove_node(move.name)
                        self.remove_node(node.name)

                        for combined_adjacent_node in combined_adjacent:
                            self.add_interference_edge(
                                coalesced_node.name, combined_adjacent_node.name
                            )
                        for combined_move_node in combined_move:
                            self.add_move_edge(
                                coalesced_node.name, combined_move_node.name
                            )
                        coalesced_nodes.append(coalesced_node)
                        break
                # Breakout of multiple loops at once: https://stackoverflow.com/a/3150107/6454085
                else:
                    # Continue if the inner loop wasn't broken.
                    continue
                # Inner loop was broken, break the outer.
                break
        return coalesced_nodes

    def freeze(self, name):
        """Freezes a node in the graph

        Arguments:
            name {any} -- The name of the node to freeze
        """
        node = self.get_node_by_name(name)
        node_adjacency_matrix_index = self.get_adjacency_matrix_index(name)
        for move in node.get_move():
            move_adjacency_matrix_index = self.get_adjacency_matrix_index(move.name)
            self.adjacency_matrix[move_adjacency_matrix_index][
                node_adjacency_matrix_index
            ] = 0
            self.adjacency_matrix[node_adjacency_matrix_index][
                move_adjacency_matrix_index
            ] = 0

            move.move_list.remove(node)
            move.move_list_removed.append(node)

        node.move_list_removed = node.move_list
        node.move_list = []

    def assign_registers(self) -> List[Node]:
        """Assigns registers to the nodes in the graph

        Returns:
            List[Node] -- The list of nodes and their assigned registers
        """
        self.take_snapshot()

        can_simplify_and_coalesce = True

        while can_simplify_and_coalesce:
            self.simplify()
            self.coalesce(len(self.nodes))

            # We should move on from simplification and coalescing if either:
            # the only nodes remaning are move-related or of significant degree
            can_simplify_and_coalesce = False
            for node in self.nodes.values():
                if (
                    node.count_adjacent() < len(self.registers)
                    or node.count_move() == 0
                ):
                    can_simplify_and_coalesce = True
                    break

            if not can_simplify_and_coalesce:
                move_related_nodes = [
                    node
                    for node in self.nodes.values()
                    if node.count_move() > 0
                    and node.count_adjacent() < len(self.registers)
                ]
                move_related_nodes = sorted(
                    move_related_nodes,
                    key=lambda node: node.count_adjacent(),
                    reverse=True,
                )
                if len(move_related_nodes) > 0:
                    self.freeze(move_related_nodes[0])

        if len(self.nodes) > 0:
            raise Exception(
                "Could not assign registers. Spilling is not implemented yet."
            )

        for simplified_node_id in list(self.simplified_nodes.keys()):
            if self.simplified_nodes[simplified_node_id].color == None:
                adjacent_nodes = self.simplified_nodes[
                    simplified_node_id
                ].adjacent_removed
                adjacent_colors = [node.color for node in adjacent_nodes]

                possible_registers = [
                    register
                    for register in self.registers
                    if register not in adjacent_colors
                ]
                selected_register = possible_registers[0]
                self.simplified_nodes[simplified_node_id].color = selected_register

        nodes_to_delete = []
        children_to_add: Dict[int, Node] = {}
        for node in self.simplified_nodes.values():
            if isinstance(node, CoalescedNode):
                nodes_to_delete.append(node)
                for child_node in node.nodes:
                    child_node.color = node.color
                    children_to_add[child_node.id] = child_node

        self.simplified_nodes = {
            node.id: node
            for node in self.simplified_nodes.values()
            if node not in nodes_to_delete
        } | children_to_add

        return list(self.simplified_nodes.values())


if __name__ == "__main__":
    ### Liveness Analysis ###
    # TODOs: 1) get graph visualization working - DONE
    #        2) get depth first search for node ordering working - DONE
    #        3) implement dataflow equations for liveness analysis -- IN PROGRESS
    #        4) add function to generate interference graph from liveness analysis

    control_flow_graph = DirectedGraph()
    control_flow_graph.add_temporary("a")
    control_flow_graph.add_temporary("b")
    control_flow_graph.add_temporary("c")

    control_flow_graph.add_node("a = 0", defines=["a"], start_node=True)
    control_flow_graph.add_node("b = a+1", uses=["a"], defines=["b"])
    control_flow_graph.add_node("c = c+b", uses=["b", "c"], defines=["c"])
    control_flow_graph.add_node("a = b*2", uses=["b"], defines=["a"])
    control_flow_graph.add_node("a<N", uses=["a"])
    control_flow_graph.add_node("return c", uses=["c"])

    control_flow_graph.add_edge("a = 0", "b = a+1")
    control_flow_graph.add_edge("b = a+1", "c = c+b")
    control_flow_graph.add_edge("c = c+b", "a = b*2")
    control_flow_graph.add_edge("a = b*2", "a<N")
    control_flow_graph.add_edge("a<N", "b = a+1")
    control_flow_graph.add_edge("a<N", "return c")

    control_flow_graph.calculate_liveness()
    print(control_flow_graph)

    ### Interference Graph Testing ###

    interference_graph = InterferenceGraph([1, 2, 3, 4])

    interference_graph.add_node("b")
    interference_graph.add_node("c")
    interference_graph.add_node("d")
    interference_graph.add_node("e")
    interference_graph.add_node("f")
    interference_graph.add_node("g")
    interference_graph.add_node("h")
    interference_graph.add_node("j")
    interference_graph.add_node("k")
    interference_graph.add_node("m")

    interference_graph.add_interference_edge("b", "c")
    interference_graph.add_interference_edge("b", "d")
    interference_graph.add_interference_edge("b", "e")
    interference_graph.add_interference_edge("b", "k")
    interference_graph.add_interference_edge("b", "m")
    interference_graph.add_move_edge("b", "j")

    interference_graph.add_interference_edge("c", "m")
    interference_graph.add_move_edge("c", "d")

    interference_graph.add_interference_edge("d", "k")
    interference_graph.add_interference_edge("d", "j")
    interference_graph.add_interference_edge("d", "m")

    interference_graph.add_interference_edge("e", "f")
    interference_graph.add_interference_edge("e", "j")
    interference_graph.add_interference_edge("e", "m")

    interference_graph.add_interference_edge("f", "j")
    interference_graph.add_interference_edge("f", "m")

    interference_graph.add_interference_edge("g", "k")
    interference_graph.add_interference_edge("g", "h")
    interference_graph.add_interference_edge("g", "j")

    interference_graph.add_interference_edge("h", "j")

    interference_graph.add_interference_edge("j", "k")

    print(interference_graph)

    print(interference_graph.assign_registers())
