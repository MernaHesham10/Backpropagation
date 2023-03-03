### Importing Packages ###
from tkinter import *
import tkinter as tk
from tkinter.ttk import Combobox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Features => bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, gender, species

# Creating Object From Tkinter
BackTkinterWindow = Tk()


def ProcessingUserDataInput(numberOfHidden, numberOfNeurons, learningRate, numberOfEpochs, activationFunction, addBias):


    def PreprocessingLoadedDataCsvFile():

        loadedDataPenguins = pd.read_csv("penguins.csv")

        # Handlign NaN Value That Found In Gender Column
        loadedDataPenguins["gender"].fillna("Not Identified", inplace=True)

        # label Encoding For Gender Column
        loadedDataPenguins.replace({"gender": {'male': 2, 'female': 1, 'Not Identified': 0}}, inplace=True)

        loadedDataPenguins = loadedDataPenguins.sample(frac=1).reset_index(drop=True)  # Shuffle

        loadedDataPenguins_X = loadedDataPenguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g']]
        loadedDataPenguins_X = np.array(loadedDataPenguins_X)

        loadedDataPenguins_encoder = OneHotEncoder(sparse=False)

        loadedDataPenguins_Y = loadedDataPenguins.species
        loadedDataPenguins_Y = loadedDataPenguins_encoder.fit_transform(np.array(loadedDataPenguins_Y).reshape(-1, 1))

        loadedDataPenguins_X_train, loadedDataPenguins_X_test, loadedDataPenguins_Y_train, loadedDataPenguins_Y_test = train_test_split(loadedDataPenguins_X, loadedDataPenguins_Y, test_size=0.4)

        loadedDataPenguins_X_train, loadedDataPenguins_X_val, loadedDataPenguins_Y_train, loadedDataPenguins_Y_val = train_test_split(loadedDataPenguins_X_train, loadedDataPenguins_Y_train, test_size=0.4)

        return loadedDataPenguins_X, loadedDataPenguins_Y, loadedDataPenguins_X_train, loadedDataPenguins_X_test, loadedDataPenguins_Y_train, loadedDataPenguins_Y_test, loadedDataPenguins_X_train, loadedDataPenguins_X_val, loadedDataPenguins_Y_train, loadedDataPenguins_Y_val


    def NeuralNetwork(addBias, activationFunction, numberOfHidden, loadedDataPenguins_X_train, loadedDataPenguins_Y_train, loadedDataPenguins_X_val=None, loadedDataPenguins_Y_val=None, numberOfEpochs = numberOfEpochs, nodes=[], learningRate = learningRate):
        numberOfHidden = len(nodes) - 1

        weightsValues = CreatngInitialWeightsArray(addBias, nodes)

        for epochIndex in range(1, numberOfEpochs + 1):
            weightsValues = GettingTrainValues(activationFunction, loadedDataPenguins_X_train, loadedDataPenguins_Y_train, learningRate, weightsValues)

            if (epochIndex % 20 == 0):
                print("Epoch {}".format(epochIndex))
                print("Training Accuracy: {}".format(CalculatingAccuracy(activationFunction, loadedDataPenguins_X_train, loadedDataPenguins_Y_train, weightsValues)))
                if loadedDataPenguins_X_val.any():
                    print("Validation Accuracy: {}".format(CalculatingAccuracy(activationFunction, loadedDataPenguins_X_val, loadedDataPenguins_Y_val, weightsValues)))
                    print()

        return weightsValues


    def CreatngInitialWeightsArray(addBias, nodes):
        """Initialize weights with random values in [-1, 1] """
        layers, weightsValues = len(nodes), []

        for calculateLayersIndex in range(1, layers):
            calculatedWeight = [[np.random.uniform(-1, 1) for k in range(nodes[calculateLayersIndex - 1] + addBias)]
                                for savedNodes_inputValueIndex in range(nodes[calculateLayersIndex])]
            weightsValues.append(np.matrix(calculatedWeight))
        print('weightsValues = ', weightsValues)
        return weightsValues


    def CalculatingForwardPropagation(activationFunction, itemValue, weightsValues, layersValue):
        # inputItemValue => x vector
        activations, layer_input = [itemValue], itemValue

        if activationFunction == 'Sigmoid':
            for layersInputValueIndex in range(layersValue):
                activation = CalculatingSigmoidActivationFn(np.dot(layer_input, weightsValues[layersInputValueIndex].T))
                activations.append(activation)
                layer_input = np.append(1, activation)  # Augment with bias

        if activationFunction == 'Hypertan':
            for layersInputValueIndex in range(layersValue):
                activation = CalculatingTangentActivationFn(np.dot(layer_input, weightsValues[layersInputValueIndex].T))
                activations.append(activation)
                layer_input = np.append(1, activation)  # Augment with bias

        return activations


    def CalculatingBackPropagation(activationFunction, inputItemValue, inputActivationsArray, inputWeightsValues, inputLayersValue):
        calculatedOutputFinal = inputActivationsArray[-1]
        calculatedErrorValue = np.matrix(inputItemValue - calculatedOutputFinal)  # Error at output

        if activationFunction == 'Sigmoid':
            for inputLayersValueIndex in range(inputLayersValue, 0, -1):
                currActivation = inputActivationsArray[inputLayersValueIndex]

                if (inputLayersValueIndex > 1):
                    # Augment previous activation
                    prevActivation = np.append(1, inputActivationsArray[inputLayersValueIndex - 1])
                else:
                    # First hidden layer, prevActivation is input (without bias)
                    prevActivation = inputActivationsArray[0]

                calculatedDeltaValue = np.multiply(calculatedErrorValue, CalculatingDashSigmoidActivationFn(currActivation))
                inputWeightsValues[inputLayersValueIndex - 1] += learningRate * np.multiply(
                    calculatedDeltaValue.T, prevActivation)

                calculatedWeightValue = np.delete(inputWeightsValues[inputLayersValueIndex - 1], [0], axis=1)  # Remove bias from weights
                calculatedErrorValue = np.dot(calculatedDeltaValue, calculatedWeightValue)  # Calculate error for current layer

        if activationFunction == 'Hypertan':
            for inputLayersValueIndex in range(inputLayersValue, 0, -1):
                currActivation = inputActivationsArray[inputLayersValueIndex]

                if (inputLayersValueIndex > 1):
                    # Augment previous activation
                    prevActivation = np.append(1, inputActivationsArray[inputLayersValueIndex - 1])
                else:
                    # First hidden layer, prevActivation is input (without bias)
                    prevActivation = inputActivationsArray[0]

                calculatedDeltaValue = np.multiply(calculatedErrorValue,
                                                   CalculatingDashTangentActivationFn(currActivation))
                inputWeightsValues[inputLayersValueIndex - 1] += learningRate * np.multiply(
                    calculatedDeltaValue.T, prevActivation)

                calculatedWeightValue = np.delete(inputWeightsValues[inputLayersValueIndex - 1], [0],
                                                  axis=1)  # Remove bias from weights
                calculatedErrorValue = np.dot(calculatedDeltaValue,
                                              calculatedWeightValue)  # Calculate error for current layer

        return inputWeightsValues


    def GettingTrainValues(activationFunction, inputX_Value, inputY_Value, learningRate, inputWeightsValues):
        inputLayersValues = len(inputWeightsValues)

        for inputData_X_Index in range(len(inputX_Value)):
            inputX_ValueTemp, inputY_ValueTemp = inputX_Value[inputData_X_Index], inputY_Value[inputData_X_Index]
            inputX_ValueTemp = np.matrix(np.append(1, inputX_ValueTemp))  # Augment feature vector

            activations = CalculatingForwardPropagation(activationFunction, inputX_ValueTemp, inputWeightsValues, inputLayersValues)
            inputWeightsValues = CalculatingBackPropagation(activationFunction, inputY_ValueTemp, activations, inputWeightsValues, inputLayersValues)

        return inputWeightsValues


    def CalculatingSigmoidActivationFn(inputValue):
        return 1 / (1 + np.exp(-inputValue))


    def CalculatingDashSigmoidActivationFn(inputValue):
        return np.multiply(inputValue, 1 - inputValue)


    def CalculatingTangentActivationFn(inputValue):
        return (1 - np.exp(-inputValue)) / (1 + np.exp(-inputValue))



    def CalculatingDashTangentActivationFn(inputValue):
        return np.multiply((1 - inputValue), (1 + inputValue))


    def CalculatingPredictSpecies(activationFunction, inputItemValue, inputWeightsValues):
        calculatedLayersValue = len(inputWeightsValues)
        inputItemValue = np.append(1, inputItemValue)  # Augment feature vector

        ##_Forward Propagation_##
        forward = CalculatingForwardPropagation(activationFunction, inputItemValue, inputWeightsValues, calculatedLayersValue)

        output = forward[-1].A1
        maxActivation = CalculatingMaxActivation(output)

        # Initialize prediction vector to zeros
        predictionArray = [0 for i in range(len(output))]
        predictionArray[maxActivation] = 1  # Set guessed class to 1

        return predictionArray  # Return prediction vector


    def CalculatingMaxActivation(outputValue):
        """Find max activation in output"""
        firstOutputValue, calculateIndex = outputValue[0], 0

        for outputValueIndex in range(1, len(outputValue)):
            if (outputValue[outputValueIndex] > firstOutputValue):
                firstOutputValue, calculateIndex = outputValue[outputValueIndex], outputValueIndex

        return calculateIndex


    def CalculatingAccuracy(activationFunction, inputX_Value, inputY_Value, inputWeightsValues):
        """Run set through network, find overall accuracy"""
        calculateCorrectGuessValue = 0
        caluculateC1Right = 0
        caluculateC1Wrong = 0
        caluculateC2Right = 0
        caluculateC2Wrong = 0
        caluculateC3Right = 0
        caluculateC3Wrong = 0

        for i in range(len(inputX_Value)):
            inputX_ListValue, inputY_ListValue = inputX_Value[i], list(inputY_Value[i])
            predictSpecies = CalculatingPredictSpecies(activationFunction, inputX_ListValue, inputWeightsValues)

            if (inputY_ListValue == predictSpecies) and np.array_equal(inputY_ListValue, ([1, 0, 0])):
                # Guessed correctly
                calculateCorrectGuessValue += 1
                caluculateC1Right += 1

            elif (inputY_ListValue == predictSpecies) and np.array_equal(inputY_ListValue, ([0, 1, 0])):
                calculateCorrectGuessValue += 1
                caluculateC2Right += 1

            elif (inputY_ListValue == predictSpecies) and np.array_equal(inputY_ListValue, ([0, 0, 1])):
                calculateCorrectGuessValue += 1
                caluculateC3Right += 1

            elif (inputY_ListValue != predictSpecies) and np.array_equal(inputY_ListValue, ([1, 0, 0])):
                caluculateC1Wrong += 1

            elif (inputY_ListValue != predictSpecies) and np.array_equal(inputY_ListValue , ([0, 1, 0])):
                caluculateC1Wrong += 1

            elif (inputY_ListValue != predictSpecies) and np.array_equal(inputY_ListValue , ([0, 0, 1])):
                caluculateC1Wrong += 1

        confusionMatrix = np.empty([3, 2])
        confusionMatrix[0][0] = caluculateC1Right
        confusionMatrix[0][1] = caluculateC1Wrong
        confusionMatrix[1][0] = caluculateC2Right
        confusionMatrix[1][1] = caluculateC2Wrong
        confusionMatrix[2][0] = caluculateC3Right
        confusionMatrix[2][1] = caluculateC3Wrong

        return (calculateCorrectGuessValue / len(inputX_Value)) * 100, confusionMatrix


    def main(numberOfHidden, numberOfNeurons, learningRate, numberOfEpochs, activationFunction, addBias):
        data_X, data_Y, data_X_train, data_X_test, data_Y_train, data_Y_test, data_X_train, data_X_val, data_Y_train, data_Y_val = PreprocessingLoadedDataCsvFile()

        data_features = len(data_X[0])  # Number of features
        data_Species = len(data_Y[0])  # Number of classes

        data_numberOfLayers = [data_features, numberOfNeurons, data_Species]  # Number of nodes in data_numberOfLayers

        calculatedWeightsValues = NeuralNetwork(addBias, activationFunction, numberOfHidden, data_X_train, data_Y_train, data_X_val, data_Y_val, numberOfEpochs=numberOfEpochs, nodes=data_numberOfLayers, learningRate=learningRate)
        print("Testing Accuracy: {}".format(CalculatingAccuracy(activationFunction, data_X_test, data_Y_test, calculatedWeightsValues)))

        accuracy, confusionMatrix = CalculatingAccuracy(activationFunction, data_X_test, data_Y_test, calculatedWeightsValues)
        DisplayingProcessDataInput(accuracy, confusionMatrix)

    main(numberOfHidden, numberOfNeurons, learningRate, numberOfEpochs, activationFunction, addBias)


# This Is Function to Clear All Content That User Entered
def ClearAllUserDataInput():

    activationFunction_Combox.set('Sigmoid')

    learningRate_Entry.delete(0, END)
    learningRate_Entry.insert(END, 0)

    numberOfEpochs_Entry.delete(0, END)
    numberOfEpochs_Entry.insert(END, 0)

    numberOfHidden_Entry.delete(0, END)
    numberOfHidden_Entry.insert(END, 0)

    numberOfNeurons_Entry.delete(0, END)
    numberOfNeurons_Entry.insert(END, 0)

    addBiasValue_Checkbox.deselect()

    displayingProcessDataInput_Label['text'] = ''


# This Is Function To Take All Data Values That User Entered
def GettingAllUserDataInput():

    savedActivationFunction = activationFunction_Tuple.index(str(activationFunction.get()))

    savedLearningRate = float(learningRate.get())

    savedNumberOfEpochs = int(numberOfEpochs.get())

    savedNumberOfHidden = int(numberOfHidden.get())

    savedNumberOfNeurons = int(numberOfNeurons.get())


    savedAddBias = addBias.get()

    print(savedNumberOfHidden, savedNumberOfNeurons, savedLearningRate, savedNumberOfEpochs, activationFunction_Tuple[savedActivationFunction], savedAddBias)

    ProcessingUserDataInput(savedNumberOfHidden, savedNumberOfNeurons, savedLearningRate, savedNumberOfEpochs, activationFunction_Tuple[savedActivationFunction], savedAddBias)


def DisplayingProcessDataInput(calculateAcuuracyValue, calculateConfusionMatrix_Array):
    displayingProcessDataInput_Label["text"] = f"Output \n Accuracy =  {int(calculateAcuuracyValue)} % \n Confusion Matrix \n {calculateConfusionMatrix_Array}"


def DataPreprocessing():
    # Loading Penguins DataSet
    penguinsLoadedData = pd.read_csv('penguins.csv')

    # Handlign NaN Value That Found In Gender Column
    penguinsLoadedData["gender"].fillna("Not Identified", inplace=True)

    # label Encoding For Species Column
    #penguinsLoadedData.replace({"species": {'Adelie': 0, 'Gentoo': 1, 'Chinstrap': 2}}, inplace=True)

    # label Encoding For Gender Column
    penguinsLoadedData.replace({"gender": {'male': 2, 'female': 1, 'Not Identified': 0}}, inplace=True)

    return penguinsLoadedData


# This Is Function To Make Whole DataSet Plots
def MakingWholeDataSetPlots():

    penguinsLoadedData = DataPreprocessing()

    # Dividing DataSet Into 3 Classes According to their Species
    data_Grouped_1 = penguinsLoadedData.groupby(penguinsLoadedData.species)
    data_AdelieGroup = data_Grouped_1.get_group('Adelie')

    data_Grouped_2 = penguinsLoadedData.groupby(penguinsLoadedData.species)
    data_GentooGroup = data_Grouped_2.get_group('Gentoo')

    data_Grouped_3 = penguinsLoadedData.groupby(penguinsLoadedData.species)
    data_ChinstrapGroup = data_Grouped_3.get_group('Chinstrap')

    plt.figure('Figure bill_length_mm VS bill_depth_mm Features')
    plt.title("bill_length_mm VS bill_depth_mm Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['bill_length_mm'], data_AdelieGroup['bill_depth_mm'])
    plt.scatter(data_ChinstrapGroup['bill_length_mm'], data_ChinstrapGroup['bill_depth_mm'])
    plt.scatter(data_GentooGroup['bill_length_mm'], data_GentooGroup['bill_depth_mm'])
    plt.xlabel('bill_length_mm')
    plt.ylabel('bill_depth_mm')
    # plt.savefig("bill_length_mm vs bill_depth_mm Figure .jpg")
    plt.show()

    plt.figure('Figure bill_length_mm VS flipper_length_mm Features')
    plt.title("bill_length_mm VS flipper_length_mm Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['bill_length_mm'], data_AdelieGroup['flipper_length_mm'])
    plt.scatter(data_ChinstrapGroup['bill_length_mm'], data_ChinstrapGroup['flipper_length_mm'])
    plt.scatter(data_GentooGroup['bill_length_mm'], data_GentooGroup['flipper_length_mm'])
    plt.xlabel('bill_length_mm')
    plt.ylabel('flipper_length_mm')
    # plt.savefig("bill_length_mm vs flipper_length_mm .jpg")
    plt.show()

    plt.figure('Figure bill_length_mm VS body_mass_g Features')
    plt.title("bill_length_mm VS body_mass_g Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['bill_length_mm'], data_AdelieGroup['body_mass_g'])
    plt.scatter(data_ChinstrapGroup['bill_length_mm'], data_ChinstrapGroup['body_mass_g'])
    plt.scatter(data_GentooGroup['bill_length_mm'], data_GentooGroup['body_mass_g'])
    plt.xlabel('bill_length_mm')
    plt.ylabel('body_mass_g')
    # plt.savefig("bill_length_mm vs body_mass_g Figure .jpg")
    plt.show()

    plt.figure('Figure bill_length_mm VS gender Features')
    plt.title("bill_length_mm VS gender Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['bill_length_mm'], data_AdelieGroup['gender'])
    plt.scatter(data_ChinstrapGroup['bill_length_mm'], data_ChinstrapGroup['gender'])
    plt.scatter(data_GentooGroup['bill_length_mm'], data_GentooGroup['gender'])
    plt.xlabel('bill_length_mm')
    plt.ylabel('gender')
    # plt.savefig("bill_length_mm vs gender Figure .jpg")
    plt.show()

    plt.figure('Figure bill_depth_mm VS flipper_length_mm Features')
    plt.title("bill_depth_mm VS flipper_length_mm Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['bill_depth_mm'], data_AdelieGroup['flipper_length_mm'])
    plt.scatter(data_ChinstrapGroup['bill_depth_mm'], data_ChinstrapGroup['flipper_length_mm'])
    plt.scatter(data_GentooGroup['bill_depth_mm'], data_GentooGroup['flipper_length_mm'])
    plt.xlabel('bill_depth_mm')
    plt.ylabel('flipper_length_mm')
    # plt.savefig("bill_depth_mm vs flipper_length_mm Figure .jpg")
    plt.show()

    plt.figure('Figure bill_depth_mm VS body_mass_g Features')
    plt.title("bill_depth_mm VS body_mass_g Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['bill_depth_mm'], data_AdelieGroup['body_mass_g'])
    plt.scatter(data_ChinstrapGroup['bill_depth_mm'], data_ChinstrapGroup['body_mass_g'])
    plt.scatter(data_GentooGroup['bill_depth_mm'], data_GentooGroup['body_mass_g'])
    plt.xlabel('bill_depth_mm')
    plt.ylabel('body_mass_g')
    # plt.savefig("bill_depth_mm vs body_mass_g Figure .jpg")

    plt.show()

    plt.figure('Figure bill_depth_mm VS gender Features')
    plt.title("bill_depth_mm VS gender Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['bill_depth_mm'], data_AdelieGroup['gender'])
    plt.scatter(data_ChinstrapGroup['bill_depth_mm'], data_ChinstrapGroup['gender'])
    plt.scatter(data_GentooGroup['bill_depth_mm'], data_GentooGroup['gender'])
    plt.xlabel('bill_depth_mm')
    plt.ylabel('gender')
    # plt.savefig("bill_depth_mm vs gender Figure .jpg")
    plt.show()

    plt.figure('Figure flipper_length_mm VS body_mass_g Features')
    plt.title("flipper_length_mm VS body_mass_g Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['flipper_length_mm'], data_AdelieGroup['body_mass_g'])
    plt.scatter(data_ChinstrapGroup['flipper_length_mm'], data_ChinstrapGroup['body_mass_g'])
    plt.scatter(data_GentooGroup['flipper_length_mm'], data_GentooGroup['body_mass_g'])
    plt.xlabel('flipper_length_mm')
    plt.ylabel('body_mass_g')
    # plt.savefig("flipper_length_mm vs body_mass_g Figure .jpg")
    plt.show()

    plt.figure('Figure flipper_length_mm VS gender Features')
    plt.title("flipper_length_mm VS gender Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['flipper_length_mm'], data_AdelieGroup['gender'])
    plt.scatter(data_ChinstrapGroup['flipper_length_mm'], data_ChinstrapGroup['gender'])
    plt.scatter(data_GentooGroup['flipper_length_mm'], data_GentooGroup['gender'])
    plt.xlabel('flipper_length_mm')
    plt.ylabel('gender')
    # plt.savefig("flipper_length_mm vs gender Figure .jpg")
    plt.show()

    plt.figure('Figure body_mass_g VS gender Features')
    plt.title("body_mass_g VS gender Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['body_mass_g'], data_AdelieGroup['gender'])
    plt.scatter(data_ChinstrapGroup['body_mass_g'], data_ChinstrapGroup['gender'])
    plt.scatter(data_GentooGroup['body_mass_g'], data_GentooGroup['gender'])
    plt.xlabel('body_mass_g')
    plt.ylabel('gender')
    # plt.savefig("body_mass_g vs gender Figure .jpg")
    plt.show()


mainTitleApp_Label = tk.Label(BackTkinterWindow, text="Back Propagation Application", font=('Lucida Calligraphy', 13), padx=20, pady=10, fg='blue')
mainTitleApp_Label.grid(column=1, row=1)

####################### Enter Number Of Hidden Layers ############################
numberOfHidden = tk.StringVar()
numberOfHidden_Label = tk.Label(BackTkinterWindow, text="Enter Number Of Hidden", font=('Lucida Calligraphy', 10), pady=10)
numberOfHidden_Entry = Entry(BackTkinterWindow, textvariable=numberOfHidden, font=('Lucida Calligraphy', 10))
numberOfHidden_Entry.insert(END, 0)

numberOfHidden_Label.grid(column=1, row=16)
numberOfHidden_Entry.grid(column=2, row=16)

##################### Enter Number Of Neurons ########################
numberOfNeurons = tk.StringVar()
numberOfNeurons_Label = tk.Label(BackTkinterWindow, text="Enter Number Of Neurons", font=('Lucida Calligraphy', 10), pady=10)
numberOfNeurons_Entry = Entry(BackTkinterWindow, textvariable=numberOfNeurons, font=('Lucida Calligraphy', 10))
numberOfNeurons_Entry.insert(END, 0)

numberOfNeurons_Label.grid(column=1, row=20)
numberOfNeurons_Entry.grid(column=2, row=20)

####################### Enter Learning Rate (eta) ############################
learningRate = tk.StringVar()
learningRate_Label = tk.Label(BackTkinterWindow, text="Enter Learning Rate (ETA)", font=('Lucida Calligraphy', 10), pady=10)
learningRate_Entry = Entry(BackTkinterWindow, textvariable=learningRate, font=('Lucida Calligraphy', 10))
learningRate_Entry.insert(END, 0)

learningRate_Label.grid(column=1, row=30)
learningRate_Entry.grid(column=2, row=30)

####################### Enter Number Of Epochs (m) ############################
numberOfEpochs = tk.StringVar()
numberOfEpochs_Label = tk.Label(BackTkinterWindow, text="Enter Number Of Epochs (m)", font=('Lucida Calligraphy', 10), pady=10)
numberOfEpochs_Entry = Entry(BackTkinterWindow, textvariable=numberOfEpochs, font=('Lucida Calligraphy', 10))
numberOfEpochs_Entry.insert(END, 0)

numberOfEpochs_Label.grid(column=1, row=53)
numberOfEpochs_Entry.grid(column=2, row=53)

##################### Select Activation Function ########################
activationFunction_Label = tk.Label(BackTkinterWindow, text="Select Activation Function", font=('Lucida Calligraphy', 10), pady=10)
activationFunction_Label.grid(column=1, row=60)

activationFunction_Tuple = ('Sigmoid', 'Hypertan')

activationFunction = StringVar()
activationFunction_Combox = Combobox(BackTkinterWindow, textvariable=activationFunction, font=('Lucida Calligraphy', 10))
activationFunction_Combox['values'] = activationFunction_Tuple
activationFunction_Combox['state'] = 'readonly'
activationFunction_Combox.set('Sigmoid')
activationFunction_Combox.grid(column= 2, row=60)

################# ADD Bias ########################
addBias = IntVar()
addBiasValue_Checkbox = Checkbutton(BackTkinterWindow, text="Add Bias", variable=addBias, font=('Lucida Calligraphy', 10), pady=10, padx=50)
addBiasValue_Checkbox.grid(column=2, row=66)

######################## Show Output Result ################################
displayingProcessDataInput_Label = tk.Label(master=BackTkinterWindow, font=('Lucida Calligraphy', 10), pady=10)
displayingProcessDataInput_Label.place(x=250, y=300)

###################### Buttons ###############################
makingPlots_Btn = Button(BackTkinterWindow, text='Show Data Plots', command=MakingWholeDataSetPlots, font=('Lucida Calligraphy', 11), pady=10, padx=50, bg='white', fg='blue')
makingPlots_Btn.place(x=210, y=450)

processData_Btn = Button(BackTkinterWindow, text='Process Inputs', command= GettingAllUserDataInput, font=('Lucida Calligraphy', 11), pady=10, padx=50, bg='white', fg='blue')
processData_Btn.place(x=220, y=520)

clearInputs_Btn = Button(BackTkinterWindow, text="Clear Inputs", command= ClearAllUserDataInput, font=('Lucida Calligraphy', 11), pady=10, padx=50, bg='white', fg='blue')
clearInputs_Btn.place(x=225, y=590)

quitApp_Btn = Button(BackTkinterWindow, text='Quit Application', command=BackTkinterWindow.quit, font=('Lucida Calligraphy', 11), pady=10, padx=50, bg='white', fg='blue')
quitApp_Btn.place(x=205, y=660)

########################### Main #############################
BackTkinterWindow.title('Back Propagation Application')
BackTkinterWindow.geometry("650x730")
BackTkinterWindow.mainloop()