import React, { useRef, useEffect, useState } from "react";
import "./styles.css";
import cv from "@techstark/opencv-js";
import * as tf from "@tensorflow/tfjs";

window.cv = cv

export default function App() {
  const [modelo, setModelo] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [facingMode, setFacingMode] = useState("user")
  const [currentStream, setCurrentStream] = useState(null)

  var video = useRef(null);
  var canvas1 = useRef(null);
  var canvas2 = useRef(null);
  var FPS = 30;
  var low_color = [25, 52, 72, 72]
  var high_color = [102, 255, 255, 255]
  var heightV = 450;
  var widthV = 650;

  function cambiarCamara() {
    console.log(currentStream)
    if (currentStream) {
      currentStream.getTracks().forEach(track => {
        track.stop();
      });
    }

    setFacingMode(facingMode === "user" ? "environment" : "user")

    var opciones = {
      audio: false,
      video: {
        facingMode: facingMode, width: widthV, height: heightV
      }
    };


    navigator.mediaDevices.getUserMedia(opciones)
      .then(function (stream) {
        setCurrentStream(stream)
        video.srcObject = currentStream;
      })
      .catch(function (err) {
        console.log("Oops, hubo un error", err);
      })
  }

  async function loadModel() {
    try {
      const model = await tf.loadLayersModel('https://apimangos.vercel.app/model.json');
      setModelo(model);
      console.log("set loaded Model");
    } catch (err) {
      console.log(err);
      console.log("failed load model");
    }
  }

  useEffect(() => {
    let constraints = {
      audio: false,
      video: {
        width: widthV,
        height: heightV,
        facingMode: facingMode
      }
    }
    if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia(constraints)
        .then(function (stream) {
          setCurrentStream(stream)
          video.current.srcObject = stream;
          procesar();
        })
        .catch(function (err0r) {
          console.log("Oops, error", err0r);
        });
    }
    tf.ready().then(() => {
      loadModel();
    });
  }, [])

  const procesar = () => {
    var src = new cv.Mat(heightV, widthV, cv.CV_8UC4)
    var cap = new cv.VideoCapture(video.current);
    var hsv = new cv.Mat();
    var blackdst = new cv.Mat();
    var dst = new cv.Mat();
    var srcResized = new cv.Mat();
    var dsize = new cv.Size(130, 90);
    let begin = Date.now();

    var context = canvas1.current.getContext('2d');
    context.drawImage(video.current, 0, 0);

    cap.read(src);
    cv.resize(src, srcResized, dsize, 0, 0, cv.INTER_AREA);
    cv.cvtColor(srcResized, hsv, cv.COLOR_BGR2HSV)

    const low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), low_color);
    const high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), high_color);

    cv.inRange(hsv, low, high, dst);

    cv.bitwise_not(dst, blackdst)

    const final = new cv.Mat();
    cv.bitwise_and(srcResized, srcResized, final, blackdst)
    cv.imshow(canvas2.current, final);

    src.delete();
    hsv.delete();
    blackdst.delete();
    dst.delete();
    srcResized.delete();
    low.delete();
    high.delete();
    final.delete();

    let delay = 1000 / FPS - (Date.now() - begin);
    setTimeout(procesar, delay);
  }

  const predecir = () => {
    var context2 = canvas2.current.getContext('2d');

    var imgData = context2.getImageData(0, 0, 130, 90);

    const example = tf.browser.fromPixels(imgData);
    example.shape = [1, 90, 130, 3]
    let class_names = ['Extra_Class', 'Class_I', 'Class_II']
    const pred = modelo.predict(example).dataSync();
    const resultado = pred.indexOf(Math.max.apply(null, pred));
    setPrediction(resultado)
    console.log({ pred, class: class_names[resultado] })
  }

  return (
    <div className='container'>
      <div className="canvasContainer">
        <video id="video" playsInline autoPlay style={{ display: 'none' }} width="650" height="450" ref={video}></video>
        <canvas id="canvas" width="650" height="450" ref={canvas1}></canvas>
        <canvas id="canvas2" width="130" height="90" ref={canvas2}></canvas>
      </div>
      <div className="optionsContainer">
        <button onClick={predecir}>predecir</button>
        <button onClick={cambiarCamara}>Cambiar camara</button>
        <div className="resultContainer">
          <h1>Prediccion:</h1>
          <div className={`extraClass${prediction === 0 ? ' on' : ''}`}>
            Extra Class
          </div>
          <div className={`classI${prediction === 1 ? ' on' : ''}`}>
            Class I
          </div>
          <div className={`classII${prediction === 2 ? ' on' : ''}`}>
            Class II
          </div>
        </div>
      </div>
    </div>
  );
}
