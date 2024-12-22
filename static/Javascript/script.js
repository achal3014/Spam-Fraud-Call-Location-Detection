function toggleConversationInput(option) {
  const audioInput = document.getElementById('audioInput');
  const textInput = document.getElementById('textInput');
  
  if (option === 'audio') {
      audioInput.classList.remove('hidden');
      textInput.classList.add('hidden');
      textInput.value = '';
  } else if (option === 'text') {
      textInput.classList.remove('hidden');
      audioInput.classList.add('hidden');
      audioInput.value = '';
  }
}

document.getElementById('callReportForm').addEventListener('submit', async function (event) {
  event.preventDefault();

  const phoneNumber = document.getElementById('phoneNumber').value;
  const audioInput = document.getElementById('audioInput').files;
  const textInput = document.getElementById('textInput').value.trim();

  if (audioInput.length === 0 && textInput === "") {
      console.error("Please provide either an audio file or a text conversation.");
      return;
  }

  const formData = new FormData();
  formData.append('phoneNumber', phoneNumber);
  formData.append('callTimings', document.getElementById('callTimings').value);
  formData.append('callDuration', document.getElementById('callDuration').value);
  formData.append('frequencyPerDay', document.getElementById('frequencyPerDay').value);
  formData.append('frequencyPerWeek', document.getElementById('frequencyPerWeek').value);

  if (audioInput.length > 0) {
      if (!audioInput[0].type.startsWith('audio/')) {
          console.error("Please upload a valid audio file.");
          return;
      }
      formData.append('audio', audioInput[0]);
  } else {
      formData.append('conversationText', textInput);
  }

  try {
      const transcribeResponse = await fetch('http://localhost:3000/transcribe-audio', {
          method: 'POST',
          body: formData,
      });

      const transcribeResult = await transcribeResponse.json();

      if (transcribeResponse.ok) {
          console.log('Prediction:', transcribeResult.prediction);
          
          const predictionElement = document.getElementById('predictionResult');
          predictionElement.textContent = `Prediction: ${transcribeResult.prediction}`;

          await fetchLocationAndDisplayMap(phoneNumber);
      } else {
          console.error('Error:', transcribeResult.error || 'Something went wrong!');
      }
  } catch (error) {
      console.error('Error submitting form:', error);
  }

  this.reset();
  document.getElementById('audioInput').classList.add('hidden');
  document.getElementById('textInput').classList.add('hidden');
});

async function fetchLocationAndDisplayMap(phoneNumber) {
  try {
      const locationResponse = await fetch('http://localhost:3000/get-location', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json'
          },
          body: JSON.stringify({ phone_number: phoneNumber })
      });

      const locationData = await locationResponse.json();
      const mapContainer = document.getElementById('mapContainer');
      const map = document.getElementById('map');
      const location= document.getElementById('location');

      if (locationResponse.ok) {
          const { latitude, longitude, city } = locationData;
          map.src = `https://www.google.com/maps?q=${latitude},${longitude}&hl=en&z=14&output=embed`;
          mapContainer.classList.remove('hidden');
          console.log(`Location detected: ${city}`);
          location.textContent=`location of the call: ${city}`
      } else {
          mapContainer.classList.add('hidden');
          console.error(locationData.error || 'Failed to fetch location.');
      }
  } catch (error) {
      console.error('Error fetching location:', error);
  }
}