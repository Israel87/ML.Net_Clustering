using Microsoft.ML;
using System;
using System.IO;

namespace ClusterML
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 0);
            IDataView dataView = mlContext.Data.LoadFromTextFile<IrisData>(_dataPath, hasHeader: false, separatorChar: ',');

            string featuresColumnName = "Features";
            var pipeline = mlContext.Transforms.Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));

            var model = pipeline.Fit(dataView);

            // previewing the pipeline...
            var preview = pipeline.Preview(dataView, 100, 100);
            Console.WriteLine(preview);

            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                try
                {
                    mlContext.Model.Save(model, dataView.Schema, fileStream);
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message.ToString());
                }
              
            }

            var predictor = mlContext.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);

            var prediction = predictor.Predict(TestIrisData.Setosa);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
        }
    }
}
