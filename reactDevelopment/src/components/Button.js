

const Button = ( {text, func} ) => {
  return (
    <button onClick={func} className='button'> {text} </button>
  )
}

export default Button