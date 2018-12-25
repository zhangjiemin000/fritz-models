import CreateMLUI
import CreateML
import Foundation

let dataFilename = "/Users/jltoole/fritz/fritz-createml-examples/subreddit_title_classifier/popular_data_top_year.json"
let data = try MLDataTable(contentsOf: URL(fileURLWithPath: dataFilename))
print(data.description)

let (trainingData, testingData) = data.randomSplit(by: 0.8, seed: 5)

let subredditClassifier = try MLTextClassifier(trainingData: trainingData,
                                               textColumn: "text",
                                               labelColumn: "label")

// Training accuracy as a percentage
let trainingAccuracy = (1.0 - subredditClassifier.trainingMetrics.classificationError) * 100
// Validation accuracy as a percentage
let validationAccuracy = (1.0 - subredditClassifier.validationMetrics.classificationError) * 100
print("Training Accuracy: \(trainingAccuracy), Validation Accuracy: \(validationAccuracy)")

let evaluationMetrics = subredditClassifier.evaluation(on: testingData)

// Evaluation accuracy as a percentage
let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100
print("Evaluation Accuracy: \(evaluationAccuracy)")

let title = "Saw this good boy at the park today with TensorFlow."
let predictedSubreddit = try subredditClassifier.prediction(from: title)
print(predictedSubreddit)

let metadata = MLModelMetadata(author: "Jameson Toole",
                               shortDescription: "Predict which subreddit a post should go in based on a title.",
                               version: "1.0")

try subredditClassifier.write(to: URL(fileURLWithPath: "/Users/jltoole/fritz/fritz-createml-examples/subreddit_title_classifier/subredditClassifier.mlmodel"),
                              metadata: metadata)

testingData.
