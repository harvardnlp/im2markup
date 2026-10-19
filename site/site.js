document.querySelectorAll('[data-copy]').forEach(function (button) {
  button.hidden = false;
  button.addEventListener('click', async function () {
    var text = document.getElementById(button.dataset.copy).textContent;
    var status = document.getElementById('copy-status');
    try {
      await navigator.clipboard.writeText(text);
      status.textContent = 'BibTeX copied.';
    } catch (_) {
      var selection = window.getSelection();
      var range = document.createRange();
      range.selectNodeContents(document.getElementById(button.dataset.copy));
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = 'Citation selected. Use your browser’s Copy command.';
    }
  });
});
