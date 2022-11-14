from typing import Dict, List, Tuple
import copy


class Node:
    def __init__(self, id: int, name, color=None):
        self.id = id
        self.name = name
        self.adjacent: List[Node] = []
        self.move_list: List[Node] = []
        self.color = color

    def __repr__(self):
        if self.color:
            return f"Node({self.name}, {self.color})"
        return f"Node({self.name})"

    def get_adjacent(self):
        return self.adjacent

    def count_adjacent(self):
        return len(self.adjacent)

class CoalescedNode(Node):
    def __init__(self, id: int, node1: Node, node2: Node, color=None):
        self.node1 = node1
        self.node2 = node2
        self.adjacent: List[Node] = list(set(node1.adjacent + node2.adjacent))
        self.move_list: List[Node] = list(set(node1.move_list + node2.move_list))

    def __repr__(self):
        if self.color:
            return f"CoalescedNode(node1: {self.node1}, node2: {self.node2}, {self.color})"
        return f"CoalescedNode(node1: {self.node1}, node2: {self.node2})"

class Graph:
    def __init__(self) -> None:
        self.nodes: Dict[int, Node] = {}
        self.adjacency_matrix = []

    def __repr__(self):
        connections_output = ""

        adjacency_matrix_output = (
            "   " + " ".join([str(node) for node in self.nodes.values()]) + "\n"
        )
        for i in range(len(self.nodes.keys())):
            adjacency_matrix_output += str(self.nodes[i]) + ": "
            for j in range(len(self.nodes.keys())):
                if self.adjacency_matrix[i][j] == 1:
                    adjacency_matrix_output += "X "
                    connections_output += (
                        str(self.nodes[i]) + " - " + str(self.nodes[j]) + "\n"
                    )
                elif self.adjacency_matrix[i][j] == -1:
                    adjacency_matrix_output += "- "
                    connections_output += (
                        str(self.nodes[i]) + " = " + str(self.nodes[j]) + "\n"
                    )
                else:
                    adjacency_matrix_output += "  "
            adjacency_matrix_output += "\n"

        return connections_output + "\n" + adjacency_matrix_output

    def add_node(self, name, color=None) -> Node:
        id = len(self.nodes)
        self.nodes[id] = Node(len(self.nodes), name, color)
        for row in self.adjacency_matrix:
            row.append(0)
        self.adjacency_matrix.append([0] * len(self.nodes))

        return self.nodes[id]

    def get_node_by_name(self, name) -> Node:
        for node in self.nodes.values():
            if node.name == name:
                return node

        raise Exception("Node not found")

    def add_interference_edge(self, name1, name2) -> None:
        node1 = self.get_node_by_name(name1)
        node2 = self.get_node_by_name(name2)
        self.adjacency_matrix[node1.id][node2.id] = 1
        self.adjacency_matrix[node2.id][node1.id] = 1
        node1.adjacent.append(node2)
        node2.adjacent.append(node1)

    def add_move_edge(self, name1, name2) -> None:
        node1 = self.get_node_by_name(name1)
        node2 = self.get_node_by_name(name2)
        self.adjacency_matrix[node1.id][node2.id] = -1
        self.adjacency_matrix[node2.id][node1.id] = -1
        node1.move_list.append(node2)
        node2.move_list.append(node1)

    def check_adjacency(self, name1, name2) -> bool:
        node1 = self.get_node_by_name(name1)
        node2 = self.get_node_by_name(name2)
        return self.adjacency_matrix[node1.id][node2.id] == 1

    def get_adjacent(self, name) -> List[Node]:
        return self.get_node_by_name(name).get_adjacent()

    def get_move_list(self, name) -> List[Node]:
        return self.get_node_by_name(name).move_list

    def count_adjacent(self, name) -> int:
        return self.get_node_by_name(name).count_adjacent()


class InterferenceGraph(Graph):
    def __init__(self, registers: List) -> None:
        super().__init__()
        self.registers = registers
        self.simplified_nodes: Dict[int, Node] = {}

    def take_snapshot(self):
        self.snapshot = copy.deepcopy(self)  # type: ignore

    def remove_node(self, name) -> None:
        node = self.get_node_by_name(name)
        self.simplified_nodes[node.id] = node
        for adjacent in node.get_adjacent():
            self.adjacency_matrix[node.id][adjacent.id] = 0
            self.adjacency_matrix[adjacent.id][node.id] = 0
            if node in self.nodes[adjacent.id].adjacent:
                self.nodes[adjacent.id].adjacent.remove(node)
            if node in self.nodes[adjacent.id].move_list:
                self.nodes[adjacent.id].move_list.remove(node)
        self.nodes.pop(node.id)

    def simplify(self) -> None:
        loops_without_changes = 0
        max_loops_without_changes = len(self.nodes)

        while (
            len(
                [
                    node
                    for node in self.nodes.values()
                    if node.count_adjacent() < len(self.registers)
                ]
            )
            > 0
        ):
            simplified_node = False
            for node in list(self.nodes.values()):
                if node.count_adjacent() < len(self.registers):
                    simplified_node = True
                    self.remove_node(node.name)

            if simplified_node:
                loops_without_changes = 0
            else:
                loops_without_changes += 1

            if loops_without_changes > max_loops_without_changes:
                break

    def assign_registers(self) -> List[Node]:
        self.take_snapshot()
        self.simplify()

        if len(self.simplified_nodes) != len(self.snapshot.nodes):
            raise Exception("Could not assign registers")

        simplified_node_ids = list(self.simplified_nodes.keys())
        self.simplified_nodes[simplified_node_ids[0]].color = self.registers[0]

        for simplified_node_id in simplified_node_ids[1:]:
            adjacent_colors = [
                self.simplified_nodes[adjacent.id].color
                for adjacent in self.snapshot.nodes[
                    simplified_node_id
                ].get_adjacent()
            ]
            possible_registers = [
                register
                for register in self.registers
                if register not in adjacent_colors
            ]
            selected_register = possible_registers[0]
            self.simplified_nodes[simplified_node_id].color = selected_register

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

    interference_graph.add_interference_edge("j", "k")

    print(interference_graph.assign_registers())
