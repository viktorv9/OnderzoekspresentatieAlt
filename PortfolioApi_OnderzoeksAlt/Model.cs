
using Microsoft.ML;
using Microsoft.ML.Data;

class Model {

    static string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
    static string _imagesFolder = Path.Combine(_assetsPath, "images");
    static string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
    static string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
    static string _predictSingleImage = Path.Combine(_imagesFolder, "toaster3.jpg");
    static string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");

    static MLContext mlContext = new MLContext();

    public Model() {
        DataViewSchema predictionPipelineSchema;
        ITransformer predictionPipeline = mlContext.Model.Load("model.zip", out predictionPipelineSchema);
    }

    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;
    }

    public class ImagePrediction : ImageData
    {
        public float[] Score;

        public string PredictedLabelValue;
    }
}