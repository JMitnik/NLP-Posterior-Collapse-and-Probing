import React, { useState, useEffect, SyntheticEvent, FormEvent, useRef } from 'react';
import logo from './logo.svg';
import './App.css';

const apiUrl = 'http://127.0.0.1:5000/predict';

function App() {
  const [sentence, setSentence] = useState();
  const [prediction, setPrediction] = useState('');

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const data = { hello: 'world' };

    fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    })
      .then(res => res.json())
      .then(data => setPrediction(data.prediction));
  };

  return (
    <div className="App">
        Predict the next word!

      <form onSubmit={handleSubmit}>
        <input type="text" value={sentence} />

        <button type="submit">Predict!</button>
      </form>

      {prediction && (
        <div className="prediction-background">{prediction}</div>
      )}
    </div>
  );
}

export default App;
