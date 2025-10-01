document.addEventListener('DOMContentLoaded', () => {
    // Mobile menu functionality
    const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
    const navLinks = document.querySelector('.nav-links');

    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            // Animate hamburger menu
            const spans = mobileMenuBtn.querySelectorAll('span');
            spans.forEach(span => span.classList.toggle('active'));
        });
    }

    // Close mobile menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!mobileMenuBtn?.contains(e.target) && !navLinks?.contains(e.target)) {
            navLinks?.classList.remove('active');
            const spans = mobileMenuBtn?.querySelectorAll('span');
            spans?.forEach(span => span.classList.remove('active'));
        }
    });

    // Add fade-in animation to elements
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe all cards and sections
    document.querySelectorAll('.card, .section-title, .section-subtitle').forEach(el => {
        observer.observe(el);
    });

    // Handle video feed errors
    const videoFeed = document.querySelector('.video-feed');
    if (videoFeed) {
        videoFeed.addEventListener('error', (e) => {
            console.error('Error loading video feed:', e);
            // Show error message to user
            const errorMessage = document.createElement('div');
            errorMessage.className = 'error-message';
            errorMessage.textContent = 'Unable to load video feed. Please check your camera permissions and try again.';
            videoFeed.parentNode.appendChild(errorMessage);
        });
    }
}); 