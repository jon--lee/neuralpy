from mlp import MLP
from input_ import Input

mapping = {
    layer.type_mlp: MLP,
    layer.type_input: Input
}
