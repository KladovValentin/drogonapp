import './App.css';
//import PropTypes from 'prop-types';
import React, {useState, useEffect} from 'react';
/*import {
  Grid,
  TextField,
  Button,
  Select,
  Checkbox,
  ListItemText,
  MenuItem,
  InputLabel,
  OutlinedInput,
} from '@mui/material';*/
//import PlotlyPlot from 'react-plotlyjs';
//import Plot from 'react-plotly.js';


import PlotlyPlotComp from './components/PlotlyPlotComp';
import Button from './components/Button';
import ToggleComponent from './components/testToggle.js';


function App() {
  
  const serverAddress = "http://0.0.0.0:3000/";

  let [trainingCheckData, setTrainingCheckData] = useState();

  let [runsData, setRunsData] = useState([])
  let [predictionsData, setPredictionsData] = useState([]);
  let [targetsData, setTargetsData] = useState([]);

  let [predictionData, setPredictionData] = useState();

  let gettingPredictions = false;

  

  const fetchTrainingCheck = () => {
    var xmlHttp = new XMLHttpRequest();
    let line = "maincontroller/getVectorsForTrainingCheck"
    xmlHttp.open( "POST", serverAddress+line, false);
    xmlHttp.send( null );
    console.log(xmlHttp.responseText.length);

    let json = JSON.parse(xmlHttp.responseText);

    let trace1 = {
        x: json.runsTargets,
        y: json.valuesTargets,
        type: 'scatter'
    };

    let trace2 = {
      x: json.runsPredicted,
      y: json.valuesPredicted,
      type: 'scatter'
    };

    setTrainingCheckData([trace1,trace2])
  };

  const fetchNewPrediction = () => {
    var xmlHttp = new XMLHttpRequest();
    let line = "maincontroller/getNextContinuousPrediction"
    xmlHttp.open( "GET", serverAddress+line, false);
    xmlHttp.send( null );
    console.log("fetched new run prediction");

    let json = JSON.parse(xmlHttp.responseText);
    
    if (json.prediction > 0){
      setPredictionsData([...predictionsData, json.prediction]);
      setRunsData([...runsData, json.run]);
    }
    if (json.target > 0){
      setTargetsData([...targetsData, json.target]);
    }
    let trace1 = {
      x: runsData,
      y: predictionsData,
      type: 'scatter'
    }
    let trace2 = {
      x: runsData,
      y: targetsData,
      type: 'scatter'
    }
    setPredictionData([trace1,trace2]);
  }

  const startFetchingPredictions = () => {
    gettingPredictions = true;
    for (let i = 0; i < 1000; i++) {
    //while (gettingPredictions){
        fetchNewPrediction();
    }
  }

  const stopFetchingPredictions = () => {
    var xmlHttp = new XMLHttpRequest();
    let line = "maincontroller/stopServer"
    xmlHttp.open( "GET", serverAddress+line, false);
    xmlHttp.send( null );
    gettingPredictions = false;
  }

    return (
      <div className="App">
        
        <PlotlyPlotComp id = "trainingQualityPlot" data = {trainingCheckData} title = 'training quality check'/>
        <Button text = 'check training' func = {fetchTrainingCheck} />
        <PlotlyPlotComp id = "predictionPlot" data = {predictionData} title = "predictions"/>
        <Button text = 'start getting next run prediction' func = {startFetchingPredictions} />
        <Button text = 'stop getting next run prediction' func = {stopFetchingPredictions} />
        <ToggleComponent />
      </div>
    );
}



export default App;
