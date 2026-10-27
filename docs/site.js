'use strict';

const trajectoryCaptions = {
  time: 'Figure 2. Source and held-out trajectories, colored by time within each recording.',
  behavior: 'Figure A4. Source and held-out trajectories, colored by behavior label within each recording.',
  stimulus: 'Figure A5. Source and held-out trajectories, colored by stimulus label within each recording.'
};
function updateTrajectory(key) {
  const src = `assets/trajectories-${key}.webp`;
  const img = document.getElementById('trajectory-image');
  img.src = src;
  img.alt = `Latent trajectories colored by ${key} for source fish 59 and 63 and three held-out fish. Rows compare identity-guided routing, CEBRA, MARBLE, LFADS, and Joint PCA.`;
  document.getElementById('trajectory-link').href = src;
  document.getElementById('trajectory-caption').textContent = trajectoryCaptions[key];
}
document.getElementById('coloring').addEventListener('change', event => updateTrajectory(event.target.value));

document.getElementById('copy-citation').addEventListener('click', async () => {
  const code = document.getElementById('bibtex');
  const status = document.getElementById('copy-status');
  const button = document.getElementById('copy-citation');
  try {
    await navigator.clipboard.writeText(code.textContent);
    button.textContent = 'Copied';
    status.textContent = '';
    setTimeout(() => { button.textContent = 'Copy'; }, 2200);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(code);
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or Command+C to copy.';
    button.textContent = 'Selected — press Ctrl/Cmd+C';
  }
});

updateTrajectory(document.getElementById('coloring').value);
