# Vital Partnerships

## Machine Learning System

Vital has a real-time prediction microservice called `predictor`. Predictor loads neural network and other machine learning models into memory on boot up. As patient data like vital signs, free-text notes, and lab results change, those changes are pushed to `predictor` for an updated prediction.

## JSON API

Predictor receives JSON requests that look like:
```
{
    "models": [{"id": "b31ea9e8-1ec9-48ec-806d-7e5a5cc94fad", "version": 1}],
    "payload": {
      "Age in Days": 13505,
      "Reason for Visit": "Headache and cough.",
      "BUN/Creatnine": 12.34,
      "BP Systolic": 120
    }
}
```

The `models` tells us which model ID to predict against, and which version of that model. The model itself receives just key-value pairs.

Vital will send the following variables from the EHR, with each prediction, if they are available:
```
    object Demo extends Enumeration {
      val AgeInDays = Value("Age in Days")
      val InsuranceType = Value("Insurance Payer Type")
      val Race = Value("Race")
      val Residence = Value("Residence")
      val Sex = Value("Sex")
    }

    object History extends Enumeration {
      val MostRecentProblem = Value("Most Recent Problem")
      val NumPriorVisits = Value("Number of Prior ED Visits")
      val NumProblems = Value("Number of Problems")
      val Problems = Value("Problems")
      val VisitLast30 = Value("ED Visit in Last 30 Days")
      val VisitLast7 = Value("ED Visit in Last 7 Days")
    }

    object Notes extends Enumeration {
      val DocAssess = Value("Doctor Assessment Note")
      val DocDispo = Value("Doctor Disposition Note")
      val NurseAssess = Value("Nurse Assessment Note")
      val Other = Value("Other Note")
      val Triage = Value("Triage Note")
    }

    object Orders extends Enumeration {
      val Imaging = Value("Imaging Orders")
      val Labs = Value("Lab Orders")
      val Meds = Value("Medications Given")
      val NumImaging = Value("Number of Imaging Orders")
      val NumLabs = Value("Number of Lab Orders")
      val NumMeds = Value("Number of Medications Given")
      val NumProcedures = Value("Number of Procedures")
      val Procedures = Value("Procedures")
    }

    object Visit extends Enumeration {
      val ArrivalMode = Value("Arrival Mode")
      val ChiefComplaint = Value("Chief Complaint")
      val ESI = Value("Emergency Severity")
      val Facility = Value("Facility")
      val IsInjury = Value("Is Injury")
      val ReasonForVisit = Value("Reason for Visit")
    }

    object Vitals extends Enumeration {
      val BpDiastolic = Value("BP Diastolic")
      val BpSystolic = Value("BP Systolic")
      val HeartRate = Value("Heart Rate")
      val Oximetry = Value("Oximetry")
      val Pain = Value("Pain 1-10")
      val RespiratoryRate = Value("Respiratory Rate")
      val TempC = Value("Temperature (C)")
    }

    // All in Minutes
    object Wait extends Enumeration {
      val BedAssigned = Value("Bed Assignment")
      val DocAssess = Value("MD Assessment")
      val DocDone = Value("MD Done")
      val LengthOfVisit = Value("Length of Visit")
      val Other = Value("Other")
      val Triage = Value("Triage")
    }
```
Inputs to your model must match our names exactly (case insensitive). If, for example, you have an input expecting the string "Blood Pressure Systolic" and we send `"BP Systolic" == 120` you may miss this input.

### Model Specification

Vital's system is built on [DL4J](https://deeplearning4j.org/), and needs either:
1) A pre-trained model exported using Keras, which supports Tensorflow, CNTK and Theano backends. See https://deeplearning4j.org/docs/latest/keras-import-overview
or
2) A model configuration file and CSV training data

Model configuration in DL4J looks nearly the same as in any other system (Torch, Tensorflow) and we can help you translate:

*Logistic Regression*
```
    new NeuralNetConfiguration.Builder()
      .seed(1234)
      .updater(new SGD(1e-2))
      .weightInit(WeightInit.XAVIER)
      .l2(1e-4)
      .list
      .layer(0, new OutputLayer.Builder().nIn(numInputs).nOut(numOutputs)
        .activation(Activation.SOFTMAX) // Soft-max, create probabilities that add to 1.0
        .lossFunction(new LossMCXENT) // Multi-class cross-entropy
        .build())
      .pretrain(false)
      .backprop(true)
      .build
```

*3-Layer Neural Network for Multi-Label Classification*
```
    new NeuralNetConfiguration.Builder()
      .seed(1234)
      .updater(new AMSGrad(1e-3))
      .l2(1e-4)
      .weightInit(WeightInit.XAVIER)
      .activation(Activation.LEAKYRELU)
      .list
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(n1).build)
      .layer(1, new DenseLayer.Builder().nIn(n1).nOut(n2).build)
      .layer(2, new OutputLayer.Builder().nIn(n2).nOut(numOutputs)
        .activation(Activation.SIGMOID)
        .lossFunction(new LossBinaryXENT())
        .build)
      .pretrain(false)
      .backprop(true)
      .build
```

*CSV Input*

Notice that the column names for inputs match the varible names provided by our API.
```
Age in Days,	Sex,	Residence,		Emergency Severity,	Chief Complaint
10220,		Male,	Private residence,	3,			Symptoms of fluid abnormalities. Discoloration or pigmentation. Symptoms referable to mouth. Symptoms referable to throat. Other symptoms referable to eye, NEC
17520,		Male,	Private residence,	4,			Shoulder symptoms. Back symptoms. Accident, NOS
```

## Model Training & Natural Language Processing

Given CSV input data and a model configuration, we would then train the file, save it securely in on Amazon S3, and then load/use the model at prediction time.

Vital supports Word2Vec, ParagraphVectors (also called Doc2Vec), and traditional feature-based NLP (words, lemmas, word-pairs, dependency parsing, etc.). If we are training the network, any text column with more than 20 distinct values will auto-matically be processed as free-text and NLP applied.

## Recurrent Neural Networks
At present, we don't have recurrent neural network models, but DL4J supports them. In this case, we will send data as follows:
```
"static": {
  "Age in Days": 13505
},
"timeSeries": {
  "BP Systolic": [
    { "value": 120, "timestamp": 1234},
    { "value": 117, "timestamp", 0123},
    ...
  ]
}
```
Essentially for any value that changes, we will provide all of the values since visit start for that variable, and the timestamp when the change occurred. In this case, no memory would need to be retained by the model itself (and retained on a per-patient basis), but it would instead do the calculation starting over from time zero each time.
