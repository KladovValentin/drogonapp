
import Plot from 'react-plotly.js';

const PlotlyPlotComp = ( {data, title} ) => {
  return (
    <Plot data={data} layout={{width: 1000, height: 500, title: {title}}}/>
  )
}


export default PlotlyPlotComp