from typing import Dict, List, Tuple
import copy


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
        return self.move_list

    def get_adjacent(self):
        return self.adjacent

    def count_adjacent(self):
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
        self.nodes.append(node)
        self.name = "-".join([str(node.name) for node in self.nodes])


class Graph:
    def __init__(self) -> None:
        self.nodes: Dict[int, Node] = {}
        self.adjacency_matrix = []

    def __repr__(self):
        connections_output = ""

        adjacency_matrix_output = (
            "   " + " ".join([str(node.name) for node in self.nodes.values()]) + "\n"
        )
        for i in self.nodes.keys():
            adjacency_matrix_output += str(self.nodes[i].name) + ": "
            i_corrected = self.get_adjacency_matrix_index(self.nodes[i].name)
            for j in self.nodes.keys():
                j_corrected = self.get_adjacency_matrix_index(self.nodes[j].name)
                if self.adjacency_matrix[i_corrected][j_corrected] == 1:
                    adjacency_matrix_output += "X "
                    connections_output += (
                        str(self.nodes[i].name) + " - " + str(self.nodes[j].name) + "\n"
                    )
                elif self.adjacency_matrix[i_corrected][j_corrected] == -1:
                    adjacency_matrix_output += "- "
                    connections_output += (
                        str(self.nodes[i].name) + " = " + str(self.nodes[j].name) + "\n"
                    )
                else:
                    adjacency_matrix_output += ". "
            adjacency_matrix_output += "\n"

        # return connections_output + "\n" + adjacency_matrix_output
        return adjacency_matrix_output

    def add_node(self, name, color=None, id=None) -> Node:
        node_id = id if id else len(self.nodes)
        node = Node(node_id, name, color)
        self.add_existing_node(node)
        return node

    def add_existing_node(self, node) -> None:
        self.nodes[node.id] = node
        for row in self.adjacency_matrix:
            row.append(0)
        self.adjacency_matrix.append([0] * len(self.nodes))

    def get_node_by_name(self, name) -> Node:
        for node in self.nodes.values():
            if node.name == name:
                return node

        raise Exception(f"Node {name} not found")

    def add_interference_edge(self, name1, name2) -> None:
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
        node1 = self.get_node_by_name(name1)
        node2 = self.get_node_by_name(name2)
        return self.adjacency_matrix[node1.id][node2.id] == 1

    def get_adjacent(self, name) -> List[Node]:
        return self.get_node_by_name(name).get_adjacent()

    def get_move_list(self, name) -> List[Node]:
        return self.get_node_by_name(name).get_move()

    def count_adjacent(self, name) -> int:
        return self.get_node_by_name(name).count_adjacent()

    def get_adjacency_matrix_index(self, name) -> int:
        return list(self.nodes.keys()).index(self.get_node_by_name(name).id)


class InterferenceGraph(Graph):
    def __init__(self, registers: List) -> None:
        super().__init__()
        self.registers = registers
        self.simplified_nodes: Dict[int, Node] = {}
        self.running_node_id = 0

    def add_node(self, name, color=None) -> Node:
        node = super().add_node(name, color, self.running_node_id)
        self.running_node_id += 1
        return node

    def take_snapshot(self):
        self.snapshot = copy.deepcopy(self)  # type: ignore

    def remove_node(self, name) -> Node:
        print(f"Removing node {name}")
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
        simplified_nodes = []
        for _ in range(num_to_simplify):
            for node in sorted(interference_graph.nodes.values(), reverse=True):
                if node.count_adjacent() < len(self.registers):
                    self.remove_node(node.name)
                    self.simplified_nodes[node.id] = node
                    simplified_nodes.append(node)
                    break
        return simplified_nodes

    def can_coalesce(self, node1: Node, node2: Node, method="briggs") -> bool:
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
        coalesced_nodes = []
        for _ in range(num_to_coalesce):
            for node in sorted(interference_graph.nodes.values(), reverse=True):
                for move in node.get_move():
                    if self.can_coalesce(node, move, method="george"):
                        print(f"\nCoalescing {node.name} and {move.name}")
                        coalesced_node = CoalescedNode(self.running_node_id, [node, move])
                        self.running_node_id += 1

                        print("before add existing")
                        print([node.name for node in self.nodes.values()])
                        self.add_existing_node(coalesced_node)

                        combined_adjacent = list(
                            set(node.get_adjacent() + move.get_adjacent())
                        )
                        combined_move = list(set(node.get_move() + move.get_move()))
                        combined_move = [node for node in combined_move if node not in [move, node]]

                        print("before remove")
                        print([node.name for node in self.nodes.values()])

                        self.remove_node(move.name)
                        print("during remove")
                        print([node.name for node in self.nodes.values()])
                        self.remove_node(node.name)
                        
                        print("after remove")
                        print([node.name for node in self.nodes.values()])

                        # print('combined adjacent:', combined_adjacent)
                        # print('combined move:', combined_move)

                        for combined_adjacent_node in combined_adjacent:
                            # print('adding edge', coalesced_node.name, combined_adjacent_node.name)
                            # print(self.nodes)
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

    def assign_registers(self) -> List[Node]:
        self.take_snapshot()
        max_attempts = len(self.nodes) * 2

        print(self)

        for _ in range(max_attempts):
            simplified_nodes = self.simplify()
            if simplified_nodes:
                print("Simplified nodes:", simplified_nodes)

            print(self)

            coalesced_nodes = self.coalesce(len(self.nodes))
            if coalesced_nodes:
                print("Coalesced nodes:", coalesced_nodes)

            if len(self.nodes) == 0:
                break

        if len(self.nodes) > 0:
            raise Exception("Could not assign registers")

        print(self.simplified_nodes)

        simplified_node_ids = list(self.simplified_nodes.keys())
        self.simplified_nodes[simplified_node_ids[0]].color = self.registers[0]

        for simplified_node_id in simplified_node_ids[1:]:
            adjacent_nodes = self.simplified_nodes[simplified_node_id].adjacent_removed
            adjacent_colors = [node.color for node in adjacent_nodes]

            print(
                self.simplified_nodes[simplified_node_id],
                adjacent_nodes,
                adjacent_colors,
            )
            # adjacent_colors = [
            #     self.simplified_nodes[adjacent.id].color
            #     for adjacent in self.snapshot.nodes[simplified_node_id].get_adjacent()
            # ]
            possible_registers = [
                register
                for register in self.registers
                if register not in adjacent_colors
            ]
            selected_register = possible_registers[0]
            self.simplified_nodes[simplified_node_id].color = selected_register

        # indices_to_delete = []
        # children_to_add: Dict[int, Node] = {}
        # for i in range(len(self.simplified_nodes)):
        #     node = self.simplified_nodes[i]
        #     if isinstance(node, CoalescedNode):
        #         indices_to_delete.append(i)
        #         for child_node in node.nodes:
        #             child_node.color = node.color
        #             children_to_add[child_node.id] = child_node

        # self.simplified_nodes = {node.id: node for node in self.simplified_nodes.values() if node.id not in indices_to_delete} | children_to_add

        return list(self.simplified_nodes.values())


if __name__ == "__main__":
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

    print(interference_graph.assign_registers())
