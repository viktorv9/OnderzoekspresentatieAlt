using Microsoft.OpenApi.Models;
using PortfolioApi;
using Microsoft.EntityFrameworkCore;
using System.Windows.Media.Imaging;

var builder = WebApplication.CreateBuilder(args);
var connectionString = builder.Configuration.GetConnectionString("Images") ?? "Data Source=Images.db";

Model model = new Model();

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
    await Task.Run(() => model.byteArrayToImage(image.Data));
    string predictedTag = await model.ClassifySingleImage();

    image.Tags = image.Tags + "," + predictedTag;

    

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

app.Run();
