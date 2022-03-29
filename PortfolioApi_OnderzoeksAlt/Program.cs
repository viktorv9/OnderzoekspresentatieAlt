using Microsoft.OpenApi.Models;
using PortfolioApi;
using Microsoft.EntityFrameworkCore;
using System.Windows.Media.Imaging;

//start code to be moved
using Microsoft.ML;
using Microsoft.ML.Data;


MLContext mlContext = new MLContext();
string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
string _imagesFolder = Path.Combine(_assetsPath, "images");
string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
string _predictSingleImage = Path.Combine(Environment.CurrentDirectory, "digi.jpg");
string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");

//end code to be moved

var builder = WebApplication.CreateBuilder(args);
var connectionString = builder.Configuration.GetConnectionString("Images") ?? "Data Source=Images.db";

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSqlite<PortfolioApi.ImageDb>(connectionString);
builder.Services.AddSwaggerGen(c =>
{
     c.SwaggerDoc("v1", new OpenApiInfo {
         Title = "Images API",
         Description = "Saving and requesting image data",
         Version = "v1" });
});

var app = builder.Build();
app.UseSwagger();
app.UseSwaggerUI(c =>
{
   c.SwaggerEndpoint("/swagger/v1/swagger.json", "Images API V1");
});

app.MapGet("/images", async (ImageDb db) => await db.Images.ToListAsync());

app.MapPost("/images", async (ImageDb db, Image image) =>
{
    await db.Images.AddAsync(image);
    await db.SaveChangesAsync();
    return Results.Created($"/images/{image.Id}", image);
});

app.MapDelete("/images/{id}", async (ImageDb db, int id) =>
{
  var image = await db.Images.FindAsync(id);
  if (image is null)
  {
    return Results.NotFound();
  }
  db.Images.Remove(image);
  await db.SaveChangesAsync();
  return Results.Ok();
});















//start code to be moved

void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
{
  foreach (ImagePrediction prediction in imagePredictionData)
  {
      Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
  }
}

void ClassifySingleImage(MLContext mlContext, ITransformer model)
{
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

}

ITransformer GenerateModel(MLContext mlContext)
{
IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
                // The image transforms transform the images into the model's expected format.
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean)).Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel).
    ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true)).Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label")).Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation")).Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
.AppendCacheCheckpoint(mlContext);



IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path:  _trainTagsTsv, hasHeader: false);
ITransformer model = pipeline.Fit(trainingData);

IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: _testTagsTsv, hasHeader: false);
IDataView predictions = model.Transform(testData);

// Create an IEnumerable for the predictions for displaying results
IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
DisplayResults(imagePredictionData);

MulticlassClassificationMetrics metrics =
    mlContext.MulticlassClassification.Evaluate(predictions,
        labelColumnName: "LabelKey",
        predictedLabelColumnName: "PredictedLabel");

        Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

return model;
}


ITransformer model = GenerateModel(mlContext);
ClassifySingleImage(mlContext, model);

//end code to be moved


app.Run();







//start code to be moved

struct InceptionSettings
{
    public const int ImageHeight = 224;
    public const int ImageWidth = 224;
    public const float Mean = 117;
    public const float Scale = 1;
    public const bool ChannelsLast = true;
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


//end code to be moved