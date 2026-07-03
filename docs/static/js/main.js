(function () {
  const figures = document.querySelectorAll('.figure-card');
  figures.forEach((figure) => {
    const img = figure.querySelector('img');
    if (!img) return;
    img.addEventListener('error', () => {
      figure.classList.add('is-missing');
    });
  });
})();
