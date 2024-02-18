
console.log("hi mainDisplay");
class MainDisplay {
    constructor() {  // Constructor
      this.shownFrame = "";
      this.listOfFrames = [];
    }
}

function switchMainDisplayToFrame(newFrame){
    document.getElementById("mainDisplay").listOfFrames.forEach(function(frame) {
        if (frame == newFrame){ 
            document.getElementById(frame).style.display = "block";
            shownFrame = newFrame;
        }
        else if (frame != newFrame){
            document.getElementById(frame).style.display = "none";
        }
    });
}

function createSplittingSideMain(){
    let splitter = document.createElement("div");
    splitter.id = "splitterSideMain";
    splitter.style.width = "4px";
    splitter.style.height = "100%";
    splitter.style.position = "absolute";
    splitter.style.backgroundColor = "#272727";
    splitter.style.left = `calc(${20}% - ${2}px)`;

    let isResizing = false;
    
    splitter.addEventListener('mousedown', (e) => {
      isResizing = true;
      document.addEventListener('mousemove', handleSplitterMouseMove);
      document.addEventListener('mouseup', () => {
        isResizing = false;
        document.removeEventListener('mousemove', handleSplitterMouseMove);
      });
    });
    document.querySelector("body").appendChild(splitter);
}

function handleSplitterMouseMove(e) {
      const leftPanelWidth = e.clientX / window.innerWidth * 100;
      const rightPanelWidth = 100 - leftPanelWidth;
  
      document.getElementById('sidePannel').style.width = `${leftPanelWidth}%`;
      document.getElementById('mainDisplay').style.width = `${rightPanelWidth}%`;
      document.getElementById('splitterSideMain').style.left = `calc(${leftPanelWidth}% - ${2}px)`;
}



document.getElementById("mainDisplay").classList.add("MainDisplay");
document.getElementById("mainDisplay").shownFrame = "thttpserver";
document.getElementById("mainDisplay").listOfFrames = ["thttpserver", "settingsPage"];  
//createSplittingSideMain();