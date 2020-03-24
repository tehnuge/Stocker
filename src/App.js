import React, { useState, useEffect } from 'react';
import logo from './logo.svg';
import './App.css';
import https from 'https';

const key = '34BY34YZCJX32Y67';
const sym = 'MSFT';
const interval = '5min';
const func = 'TIME_SERIES_INTRADAY';
const url = `https://www.alphavantage.co/query?function=${func}&symbol=${sym}&interval=${interval}&apikey=${key}`;

function App() {
  const [stocks, setStocks] = useState([]);


const processResponse = (resp) => {
  let data = '';

  // A chunk of data has been recieved.
  resp.on('data', (chunk) => {

    data += chunk;
  });

  // The whole response has been received. Print out the result.
  resp.on('end', () => {
    setStocks(data);

    //console.log(JSON.parse(data).explanation);
  });

};

  useEffect(() => {
    https.get(url, processResponse).on("error", (err) => {
    console.log('NO!')

  console.log("Error: " + err.message);
});
}, []);

  console.log(stocks);
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
