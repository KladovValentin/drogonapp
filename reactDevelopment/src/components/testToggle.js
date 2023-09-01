import React, { useState } from 'react';

const ToggleComponent = () => {
    const [isVisible, setIsVisible] = useState(true);
  
    return (
      <div>
        {isVisible && <div>This is the content of the toggle component.</div>}
        <button className='button' onClick={() => setIsVisible(!isVisible)}>Toggle Display</button>
      </div>
    );
  };
  
  export default ToggleComponent;