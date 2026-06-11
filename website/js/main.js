document.addEventListener('DOMContentLoaded', () => {
    const copyBtn = document.querySelector('.copy-btn');
    const snippet = document.querySelector('.install-snippet code');

    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            const text = snippet.innerText;
            navigator.clipboard.writeText(text).then(() => {
                copyBtn.innerHTML = '<i class="fas fa-check"></i>';
                setTimeout(() => {
                    copyBtn.innerHTML = '<i class="far fa-copy"></i>';
                }, 2000);
            });
        });
    }
});
