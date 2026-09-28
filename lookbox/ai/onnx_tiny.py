"""A tiny ONNX model, hand-encoded as protobuf (the `onnx` package isn't needed).

One Sigmoid node, input "input_image" float [1, 3, S, S] → output "output_image".
Lets the tests and `--selftest` run the *real* onnxruntime end to end without downloading anything.
"""

from __future__ import annotations


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _int(field: int, value: int) -> bytes:
    return _key(field, 0) + _varint(value)


def _bytes(field: int, payload: bytes) -> bytes:
    return _key(field, 2) + _varint(len(payload)) + payload


def _str(field: int, s: str) -> bytes:
    return _bytes(field, s.encode())


def _value_info(name: str, dims: list[int]) -> bytes:
    shape = b"".join(_bytes(1, _int(1, d)) for d in dims)  # TensorShapeProto.dim{dim_value}
    tensor = _int(1, 1) + _bytes(2, shape)  # elem_type FLOAT, shape
    return _str(1, name) + _bytes(2, _bytes(1, tensor))  # ValueInfoProto{name, TypeProto{tensor_type}}


def sigmoid_model(size: int = 16, op: str = "Sigmoid") -> bytes:
    node = _str(1, "input_image") + _str(2, "output_image") + _str(3, "n0") + _str(4, op)
    graph = (_bytes(1, node) + _str(2, "tiny")
             + _bytes(11, _value_info("input_image", [1, 3, size, size]))
             + _bytes(12, _value_info("output_image", [1, 3, size, size])))
    opset = _str(1, "") + _int(2, 13)
    return _int(1, 8) + _str(2, "lookbox-tests") + _bytes(7, graph) + _bytes(8, opset)
