
const brushWidth = 20
const color = '#000000'
let drawingcanvas

let modelConv = null
let modelSeq = null

const canvas = document.getElementById('mi-canvas')
const ctx1 = canvas.getContext('2d')
const smallcanvas = document.getElementById('smallcanvas')
const ctx2 = smallcanvas.getContext('2d')

function clean () {
  ctx1.clearRect(0, 0, canvas.width, canvas.height)
  drawingcanvas.clear()
}

function predictBoth () {
  predict(false)
  predict(true)
}

function predict (conv) {
  resampleSingle(canvas, 28, 28, smallcanvas)

  const imgData = ctx2.getImageData(0, 0, 28, 28)
  let arr = []
  let arr28 = []
  for (let p = 0, i = 0; p < imgData.data.length; p += 4) {
    const valor = imgData.data[p + 3] / 255
    arr28.push([valor])
    if (arr28.length == 28) {
      arr.push(arr28)
      arr28 = []
    }
  }

  arr = [arr]
  const tensor4 = tf.tensor4d(arr)

  const model = conv ? modelConv : modelSeq
  const results = model.predict(tensor4).dataSync()

  const prediction = results.indexOf(Math.max.apply(null, results))

  if (conv) {
    console.log('Prediccion convolucional', prediction)
    document.getElementById('resultado-convulcional').innerHTML = prediction
  } else {
    console.log('Prediccion secuencial', prediction)
    document.getElementById('resultado-secuencial').innerHTML = prediction
  }
}

function resampleSingle (canvas, width, height, resizeCanvas) {
  const width_source = canvas.width
  const height_source = canvas.height
  width = Math.round(width)
  height = Math.round(height)

  const ratio_w = width_source / width
  const ratio_h = height_source / height
  const ratio_w_half = Math.ceil(ratio_w / 2)
  const ratio_h_half = Math.ceil(ratio_h / 2)

  const ctx = canvas.getContext('2d')
  const ctx2 = resizeCanvas.getContext('2d')
  const img = ctx.getImageData(0, 0, width_source, height_source)
  const img2 = ctx2.createImageData(width, height)
  const data = img.data
  const data2 = img2.data

  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      const x2 = (i + j * width) * 4
      let weight = 0
      let weights = 0
      let weights_alpha = 0
      let gx_r = 0
      let gx_g = 0
      let gx_b = 0
      let gx_a = 0
      const center_y = (j + 0.5) * ratio_h
      const yy_start = Math.floor(j * ratio_h)
      const yy_stop = Math.ceil((j + 1) * ratio_h)
      for (let yy = yy_start; yy < yy_stop; yy++) {
        const dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half
        const center_x = (i + 0.5) * ratio_w
        const w0 = dy * dy
        const xx_start = Math.floor(i * ratio_w)
        const xx_stop = Math.ceil((i + 1) * ratio_w)
        for (let xx = xx_start; xx < xx_stop; xx++) {
          const dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half
          const w = Math.sqrt(w0 + dx * dx)
          if (w >= 1) {
            continue
          }

          weight = 2 * w * w * w - 3 * w * w + 1
          const pos_x = 4 * (xx + yy * width_source)

          gx_a += weight * data[pos_x + 3]
          weights_alpha += weight

          if (data[pos_x + 3] < 255) { weight = weight * data[pos_x + 3] / 250 }
          gx_r += weight * data[pos_x]
          gx_g += weight * data[pos_x + 1]
          gx_b += weight * data[pos_x + 2]
          weights += weight
        }
      }
      data2[x2] = gx_r / weights
      data2[x2 + 1] = gx_g / weights
      data2[x2 + 2] = gx_b / weights
      data2[x2 + 3] = gx_a / weights_alpha
    }
  }

  for (let p = 0; p < data2.length; p += 4) {
    let gris = data2[p]
    if (gris < 100) {
      gris = 0
    } else {
      gris = 255
    }
    data2[p] = gris
    data2[p + 1] = gris
    data2[p + 2] = gris
  }

  ctx2.putImageData(img2, 0, 0)
}

async function init () {
  drawingcanvas = this.__canvas = new fabric.Canvas('mi-canvas', { isDrawingMode: true })
  fabric.Object.prototype.transparentCorners = false
  if (drawingcanvas.freeDrawingBrush) {
    drawingcanvas.freeDrawingBrush.color = color
    drawingcanvas.freeDrawingBrush.width = brushWidth
  }

  modelConv = await tf.loadLayersModel('modelConv.json')
  console.log('Model has been loaded - Convolutional')

  modelSeq = await tf.loadLayersModel('modelSequential.json')
  console.log('Model has been loaded - Sequential')
}

init()
