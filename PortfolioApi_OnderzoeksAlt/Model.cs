using Microsoft.ML;
using Microsoft.ML.Data;
using System.Drawing;
using System.Windows.Media.Imaging;

class Model {

    static string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
    static string _imagesFolder = Path.Combine(_assetsPath, "images");
    static string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
    static string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
    static string _predictSingleImage = Path.Combine(Environment.CurrentDirectory, "temp.jpg");
    static string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");

    static MLContext mlContext = new MLContext();
    static ITransformer model;

    struct InceptionSettings
    {
        public const int ImageHeight = 224;
        public const int ImageWidth = 224;
        public const float Mean = 117;
        public const float Scale = 1;
        public const bool ChannelsLast = true;
    }

    public Model() {
        model = GenerateModel(mlContext);
    }

    
    private void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
    {
    foreach (ImagePrediction prediction in imagePredictionData)
    {
        Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
    }
    }

    public Task<string> ClassifySingleImage()
    {
        return Task.Run(() => {

            var imageData = new ImageData()
            {
                ImagePath = _predictSingleImage
            };

            // Make prediction function (input = ImageData, output = ImagePrediction)
            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictor.Predict(imageData);
            Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
            Console.WriteLine("=====================");

            var labelBuffer = new VBuffer<ReadOnlyMemory<char>>();
            predictor.OutputSchema["Score"].Annotations.GetValue("SlotNames", ref labelBuffer);
            var labels = labelBuffer.DenseValues().Select(l => l.ToString()).ToArray();

            var index = Array.IndexOf(labels, prediction.PredictedLabelValue);
            var score = prediction.Score[index];

            var top10scores = labels.ToDictionary(
                l => l,
                l => (decimal)prediction.Score[Array.IndexOf(labels, l)]
                )
                .OrderByDescending(kv => kv.Value)
                .Take(3);

            foreach (KeyValuePair<string, decimal> kvp in top10scores)
            {
                //textBox3.Text += ("Key = {0}, Value = {1}", kvp.Key, kvp.Value);
                Console.WriteLine("Key = {0}, Value = {1}", kvp.Key, kvp.Value);
            }

            return top10scores.First().Key;
        });

    }

    private ITransformer GenerateModel(MLContext mlContext)
    {
        IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
                    .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                    .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                    .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel).ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label")).Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation")).Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                    .AppendCacheCheckpoint(mlContext);



        IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path:  _trainTagsTsv, hasHeader: false);
        ITransformer model = pipeline.Fit(trainingData);

        IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: _testTagsTsv, hasHeader: false);
        IDataView predictions = model.Transform(testData);

        // Create an IEnumerable for the predictions for displaying results
        IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
        DisplayResults(imagePredictionData);

        return model;
    }

    public void byteArrayToImage(byte[] source)
    {
        MemoryStream ms = new MemoryStream(source);
        Image ret = Image.FromStream(ms);
        ret.Save("temp.jpg", System.Drawing.Imaging.ImageFormat.Jpeg);
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