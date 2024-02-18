//import React, {useState, useEffect} from "react";
//import Plot from 'react=plotly.js';



let serverAddress = "http://0.0.0.0:3000/";

function handleButton1Click(){
    var xmlHttp = new XMLHttpRequest();
    let line = "maincontroller/getPrediction?run1=446737374"
    xmlHttp.open( "GET", serverAddress+line, false);
    xmlHttp.send( null );
    console.log(xmlHttp.responseText);


    let voices = [];
    let synth = window.speechSynthesis;
    
    let utter = new SpeechSynthesisUtterance();
  
    utter.lang = 'en-US';
    utter.text = xmlHttp.responseText;
    utter.volume = 0.1;
    voices = window.speechSynthesis.getVoices();
    utter.voice = voices[0];
  
    synth.speak(utter);

}

function handleButton2Click(){
    var xmlHttp = new XMLHttpRequest();
    let line = "customCtrl/userName"
    xmlHttp.open( "GET", serverAddress+line, false);
    xmlHttp.send( null );
    console.log(xmlHttp.responseText);
}


function handleButton3Click(){
    var xmlHttp = new XMLHttpRequest();
    let line = "maincontroller/getVectorsForTrainingCheck"
    xmlHttp.open( "POST", serverAddress+line, false);
    xmlHttp.send( null );
    //console.log(xmlHttp.responseText);

    let json = JSON.parse(xmlHttp.responseText);
    runsTargets = json.runsTargets
    valuesTargets = json.valuesTargets
    runsPredicted = json.runsPredicted
    valuesPredicted = json.valuesPredicted

    var trace1 = {
        x: runsTargets,
        y: valuesTargets,
        type: 'scatter'
    };

    var trace2 = {
        x: runsPredicted,
        y: valuesPredicted,
        type: 'scatter'
    };

    var data = [trace1, trace2];

    Plotly.newPlot('plotlyChart', data);
}

function switchContinuousPredictionLoop(){
    var xmlHttp = new XMLHttpRequest();
    let line = "maincontroller/switchContinuousPredictionLoop"
    xmlHttp.open( "GET", serverAddress+line, false);
    xmlHttp.send( null );
    console.log(xmlHttp.responseText);

    let json = JSON.parse(xmlHttp.responseText);
    responset = json.response
}



document.getElementById("button1").addEventListener("click",function(){
    handleButton1Click();
});
document.getElementById("button2").addEventListener("click",function(){
    handleButton2Click();
});
document.getElementById("button3").addEventListener("click",function(){
    //handleButton3Click();
    switchContinuousPredictionLoop();
});

document.getElementById("settingsButton").addEventListener("click",function(){
    switchMainDisplayToFrame("settingsPage");
});
document.getElementById("thttpserverButton").addEventListener("click",function(){
    switchMainDisplayToFrame("thttpserver");
});
    