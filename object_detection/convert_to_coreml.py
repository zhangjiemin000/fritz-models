r"""Convert a tensorflow frozen graph object detection model to Core ML.
Example usage:
python convert_to_coreml.py \
--inference-graph=/path/to/export/optimized_inference_graph.pb \
--output=/path/to/export/object_detector.mlmodel
"""

import argparse
import sys

from coremltools.proto import Model_pb2
import tfcoreml


def _save_modified_mlmodel(model, filename):
    """Save modified model.
    Args:
        model (MLModel): MLModel to save
        filename (str): name of file to write to.
    """
    with open(filename, 'wb') as f:
        f.write(model.SerializeToString())


def _load_mlmodel(filename):
    """Load mlmodel from disk.
    Args:
        filename (str): name of file to load.
    Returns: model (MLModel): MLModel to modify
    """
    model = Model_pb2.Model()

    with open(filename, "rb") as f:
        model.ParseFromString(f.read())

    return model


def convert_to_coreml(inference_graph_path, output_path):
    """Convert a tensorflow inference graph to Core ML model.
    This assumes mobilenet preprocessing.
    Args:
        inference_graph_path (str): path to a tensorflow frozen graph .pb file
        output_path (str): output path for .mlmodel file
    """
    tfcoreml.convert(
        tf_model_path=inference_graph_path,
        mlmodel_path=output_path,
        output_feature_names=['concat:0', 'concat_1:0'],
        input_name_shape_dict={'Preprocessor/sub:0': [1, 300, 300, 3]},
        image_input_names='Preprocessor/sub:0',
        image_scale=2.0 / 255.0,
        red_bias=-1.0,
        green_bias=-1.0,
        blue_bias=-1.0
    )


def update_mlmodel_names(
        mlmodel_path,
        input_name,
        output_box_name,
        output_score_name):
    """Update the mlmodel file feature names.
    Args:
        mlmodel_path (str): path to a .mlmodel file
        input_name (str): desired input feature name
        output_box_name (str): desired bounding box output feature name
        output_score_name (str): desired output score feature name
    """
    model_spec = _load_mlmodel(mlmodel_path)

    # Change the descriptions
    original_input_name = model_spec.description.input[0].name
    model_spec.description.input[0].name = input_name
    original_output_names = [
        node.name for node in model_spec.description.output
    ]
    model_spec.description.output[0].name = output_score_name
    model_spec.description.output[1].name = output_box_name

    # Change the layers in the graph as well
    # input
    for k, layer in enumerate(model_spec.neuralNetwork.layers):
        if layer.input[0] == original_input_name:
            model_spec.neuralNetwork.layers[k].input[0] = input_name

    # concat_1
    for k, layer in enumerate(model_spec.neuralNetwork.layers):
        if layer.output[0] == original_output_names[0]:
            model_spec.neuralNetwork.layers[k].output[0] = output_score_name

    # concat
    for k, layer in enumerate(model_spec.neuralNetwork.layers):
        if layer.output[0] == original_output_names[1]:
            model_spec.neuralNetwork.layers[k].output[0] = output_box_name

    # Lastly fix the preprocessing
    model_spec.neuralNetwork.preprocessing[0].featureName = input_name

    _save_modified_mlmodel(model_spec, mlmodel_path)


def main(argv):
    parser = argparse.ArgumentParser(
        description='Convert Tensorflow Object Detection models to Core ML.')
    parser.add_argument(
        '--inference-graph', type=str, required=True,
        help='A tensorflow frozen graph to convert.')
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output path for the Core ML file.')
    parser.add_argument(
        '--coreml-input-name', type=str, default='image',
        help='A name for the Core ML input feature.')
    parser.add_argument(
        '--coreml-output-box-name', type=str, default='bbox_offsets',
        help='A name for the Core ML output boxes feature.')
    parser.add_argument(
        '--coreml-output-score-name', type=str, default='scores',
        help='A name for the Core ML output boxes feature.')

    args = parser.parse_args(argv)

    convert_to_coreml(args.inference_graph, args.output)
    update_mlmodel_names(
        args.output,
        args.coreml_input_name,
        args.coreml_output_box_name,
        args.coreml_output_score_name
    )


if __name__ == '__main__':
    main(sys.argv[1:])
