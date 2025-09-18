---
layout: about
title: About Me
permalink: /about/
# image: /assets/images/trp-sharpened.png
---

<h3 class="font-weight-light">Hello, I'm <span class="font-weight-bold">{{site.author.name}}</span></h3>

Welcome to my digital space! I'm a seasoned Data Scientist with a passion for crafting innovative, data-driven solutions to tackle complex business challenges. Currently, I lead a high-performance team responsible for architecting and deploying scalable data solutions that address complex business challenges. 

With extensive experience in data engineering, advanced analytics, machine learning, and deep learning, I specialize in transforming raw data into actionable insights that drive strategic decision-making and operational efficiency.

My professional journey began at Leadzpipe, where I developed end-to-end data pipelines to extract, model, and analyze Google AdWords campaign data, optimizing client performance. This work was recognized with the prestigious Aegis Graham Bell Award in 2020, underscoring my commitment to excellence in data-driven problem-solving.

With a solid foundation in Computer Science and a deep technical proficiency in data science, I am driven by a passion for leveraging cutting-edge technologies to solve intricate problems. My expertise spans data architecture, pipeline optimization, machine learning, and geospatial analytics, with a focus on delivering solutions that are not only innovative but also highly performant.

Have an unsolvable problem? I'm just an mail away – **shrikantnaidu777@gmail.com**.

---
<h3>Professional Experience</h3>

<div class="timeline-container">
    <div class="timeline">
        <div class="timeline-item left">
            <div class="timeline-content">
                <h5 class="timeline-title">Manager - Data Science & Engineering</h5>
                <p class="timeline-company">Loylty Rewardz</p>
                <p class="timeline-date">March 2024 - Present</p>
            </div>
        </div>
        <div class="timeline-item right">
            <div class="timeline-content">
                <h5 class="timeline-title">Data Scientist II</h5>
                <p class="timeline-company">Loylty Rewardz</p>
                <p class="timeline-date">March 2023 - March 2024</p>
            </div>
        </div>
        <div class="timeline-item left">
            <div class="timeline-content">
                <h5 class="timeline-title">Data Scientist</h5>
                <p class="timeline-company">Predoole Analytics</p>
                <p class="timeline-date">October 2022 - March 2023</p>
            </div>
        </div>
        <div class="timeline-item right">
            <div class="timeline-content">
                <h5 class="timeline-title">Data Analyst</h5>
                <p class="timeline-company">Medly</p>
                <p class="timeline-date">October 2020 - September 2022</p>
            </div>
        </div>
        <div class="timeline-item left">
            <div class="timeline-content">
                <h5 class="timeline-title">Data Science Intern</h5>
                <p class="timeline-company">Leadzpipe</p>
                <p class="timeline-date">October 2019 - May 2020</p>
            </div>
        </div>
    </div>
</div>
<hr>

### Awards 

<div class="image-slider">
    <div class="slider">
        <div class="slide">
            <img src="/assets/images/2-sharpened.png" alt="Image 1">
            <div class="caption">Aegis Graham Bell Awards 2020</div>
        </div>
        <div class="slide">
            <img src="/assets/images/img-2.png" alt="Image 2">
            <div class="caption">Extraordinary Diligence Award 2024</div>
        </div>
        <div class="slide">
            <img src="/assets/images/img-3.jpg" alt="Image 3">
            <div class="caption">Outside the Box Thinker Award 2025</div>
        </div>
    </div>
    <button class="prev" onclick="moveSlide(-1)">&#10094;</button>
    <button class="next" onclick="moveSlide(1)">&#10095;</button>
</div>

<style>
.image-slider {
    position: relative;
    max-width: 100%;
    margin: 2rem auto;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    padding: 1.5rem;
}

.slider {
    display: flex;
    overflow: hidden;
    border-radius: 8px;
}

.slide {
    min-width: 100%;
    transition: transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    text-align: center;
    position: relative;
}

.slide img {
    max-width: 100%;
    max-height: 450px;
    height: auto;
    object-fit: contain;
    display: block;
    margin: 0 auto;
    border-radius: 8px;
    transition: transform 0.3s ease;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.slide:hover img {
    transform: scale(1.02);
}

.caption {
    margin-top: 1.5rem;
    font-size: 1.1rem;
    font-weight: 600;
    color: #2c3e50;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    position: relative;
    padding-bottom: 0.5rem;
}

.caption::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #ff6f00, #ff8f00);
    border-radius: 2px;
}

button {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    background: rgba(255, 255, 255, 0.95);
    border: 2px solid #ff6f00;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    cursor: pointer;
    font-size: 1.2rem;
    color: #ff6f00;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    align-items: center;
    justify-content: center;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(255, 111, 0, 0.2);
}

button:hover {
    background: #ff6f00;
    color: white;
    transform: translateY(-50%) scale(1.1);
    box-shadow: 0 6px 25px rgba(255, 111, 0, 0.4);
}

button:active {
    transform: translateY(-50%) scale(0.95);
}

.prev {
    left: 20px;
}

.next {
    right: 20px;
}

/* Responsive Design */
@media (max-width: 768px) {
    .image-slider {
        margin: 1rem auto;
        padding: 1rem;
    }
    
    .slide img {
        max-height: 300px;
    }
    
    .caption {
        font-size: 1rem;
        margin-top: 1rem;
    }
    
    button {
        width: 40px;
        height: 40px;
        font-size: 1rem;
    }
    
    .prev {
        left: 10px;
    }
    
    .next {
        right: 10px;
    }
}

@media (max-width: 480px) {
    .slide img {
        max-height: 250px;
    }
    
    .caption {
        font-size: 0.9rem;
    }
    
    button {
        width: 35px;
        height: 35px;
        font-size: 0.9rem;
    }
}

/* Timeline Styles */
.timeline-container {
    max-width: 700px;
    margin: 1rem auto;
    padding: 1.5rem;
    background: #f8f9fa;
    border-radius: 6px;
    position: relative;
}

.timeline {
    position: relative;
    padding: 0.5rem 0;
}

.timeline::before {
    content: '';
    position: absolute;
    left: 50%;
    top: 0;
    bottom: 0;
    width: 2px;
    background: #6c757d;
    transform: translateX(-50%);
}

.timeline-item {
    position: relative;
    margin-bottom: 2rem;
    width: 45%;
    opacity: 0;
    transform: translateY(15px);
    animation: fadeInUp 0.4s ease forwards;
}

.timeline-item:nth-child(1) { animation-delay: 0.1s; }
.timeline-item:nth-child(2) { animation-delay: 0.15s; }
.timeline-item:nth-child(3) { animation-delay: 0.2s; }
.timeline-item:nth-child(4) { animation-delay: 0.25s; }
.timeline-item:nth-child(5) { animation-delay: 0.3s; }

@keyframes fadeInUp {
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.timeline-item.left {
    left: 0;
    text-align: right;
}

.timeline-item.right {
    left: 55%;
    text-align: left;
}

.timeline-item::before {
    content: '';
    position: absolute;
    top: 50%;
    width: 30px;
    height: 1.5px;
    background: #dee2e6;
    transform: translateY(-50%);
}

.timeline-item.left::before {
    right: -30px;
}

.timeline-item.right::before {
    left: -30px;
}

.timeline-item::after {
    content: '';
    position: absolute;
    top: 50%;
    width: 8px;
    height: 8px;
    background: #6c757d;
    border-radius: 50%;
    transform: translateY(-50%);
}

.timeline-item.left::after {
    right: -34px;
}

.timeline-item.right::after {
    left: -34px;
}

.timeline-content {
    background: #ffffff;
    padding: 1rem;
    border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
    transition: all 0.2s ease;
    position: relative;
}

.timeline-content:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.timeline-title {
    margin: 0 0 0.3rem 0;
    font-size: 1rem;
    font-weight: 600;
    color: #2c3e50;
    line-height: 1.2;
}

.timeline-company {
    margin: 0 0 0.2rem 0;
    font-size: 0.85rem;
    font-weight: 500;
    color: #495057;
}

.timeline-date {
    margin: 0;
    font-size: 0.75rem;
    color: #6c757d;
    font-weight: 400;
    font-style: italic;
}

/* Responsive Timeline */
@media (max-width: 768px) {
    .timeline-container {
        margin: 0.75rem auto;
        padding: 1.25rem;
    }
    
    .timeline::before {
        left: 1.5rem;
        transform: none;
        width: 2px;
    }
    
    .timeline-item {
        width: 100% !important;
        left: 0 !important;
        text-align: left !important;
        padding-left: 3rem;
        margin-bottom: 1.5rem;
    }
    
    .timeline-item::before {
        right: auto !important;
        left: -2.5rem !important;
        width: 2rem;
        height: 1.5px;
    }
    
    .timeline-item::after {
        right: auto !important;
        left: -2.25rem !important;
        width: 8px;
        height: 8px;
    }
    
    .timeline-content {
        padding: 0.875rem;
    }
    
    .timeline-title {
        font-size: 0.9rem;
    }
    
    .timeline-company {
        font-size: 0.8rem;
    }
    
    .timeline-date {
        font-size: 0.7rem;
    }
}

@media (max-width: 480px) {
    .timeline-container {
        padding: 1rem;
        margin: 0.5rem auto;
    }
    
    .timeline::before {
        left: 1.25rem;
        width: 1.5px;
    }
    
    .timeline-item {
        padding-left: 2.5rem;
        margin-bottom: 1.25rem;
    }
    
    .timeline-item::before {
        left: -2rem !important;
        width: 1.5rem;
        height: 1px;
    }
    
    .timeline-item::after {
        left: -1.75rem !important;
        width: 6px;
        height: 6px;
    }
    
    .timeline-content {
        padding: 0.75rem;
    }
    
    .timeline-title {
        font-size: 0.85rem;
    }
    
    .timeline-company {
        font-size: 0.75rem;
    }
    
    .timeline-date {
        font-size: 0.65rem;
    }
}

/* Socials Styles */
.socials-container {
    max-width: 600px;
    margin: 1.5rem auto;
    padding: 1.5rem;
    background: #ffffff;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.socials-grid {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-wrap: wrap;
    gap: 1.5rem;
}

.social-link {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    border-radius: 8px;
    background: #f8f9fa;
    transition: all 0.3s ease;
    text-decoration: none;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.social-link:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    background: #ffffff;
}

.social-link img {
    width: 24px;
    height: 24px;
    transition: transform 0.3s ease;
}

.social-link:hover img {
    transform: scale(1.1);
}

/* Special styling for Weights & Biases */
.social-link:nth-child(5) {
    width: auto;
    padding: 0 1rem;
    background: #ff6f00;
    color: white;
}

.social-link:nth-child(5):hover {
    background: #ff8f00;
}

.social-link:nth-child(5) img {
    width: auto;
    height: 20px;
}

/* Responsive Socials */
@media (max-width: 768px) {
    .socials-container {
        margin: 1rem auto;
        padding: 1.25rem;
    }
    
    .socials-grid {
        gap: 1rem;
    }
    
    .social-link {
        width: 44px;
        height: 44px;
    }
    
    .social-link img {
        width: 22px;
        height: 22px;
    }
    
    .social-link:nth-child(5) {
        padding: 0 0.75rem;
    }
    
    .social-link:nth-child(5) img {
        height: 18px;
    }
}

@media (max-width: 480px) {
    .socials-grid {
        gap: 0.75rem;
    }
    
    .social-link {
        width: 40px;
        height: 40px;
    }
    
    .social-link img {
        width: 20px;
        height: 20px;
    }
    
    .social-link:nth-child(5) {
        padding: 0 0.5rem;
    }
    
    .social-link:nth-child(5) img {
        height: 16px;
    }
}
</style>

<script>
let currentSlide = 0;
let autoPlayInterval;
let isHovered = false;

function showSlide(index) {
    const slides = document.querySelectorAll('.slide');
    if (index >= slides.length) {
        currentSlide = 0;
    } else if (index < 0) {
        currentSlide = slides.length - 1;
    } else {
        currentSlide = index;
    }
    const offset = -currentSlide * 100;
    slides.forEach(slide => {
        slide.style.transform = `translateX(${offset}%)`;
    });
}

function moveSlide(direction) {
    showSlide(currentSlide + direction);
    resetAutoPlay();
}

function startAutoPlay() {
    autoPlayInterval = setInterval(() => {
        if (!isHovered) {
            moveSlide(1);
        }
    }, 5000); // Auto-advance every 5 seconds
}

function stopAutoPlay() {
    clearInterval(autoPlayInterval);
}

function resetAutoPlay() {
    stopAutoPlay();
    startAutoPlay();
}

// Initialize the slider
document.addEventListener('DOMContentLoaded', function() {
    showSlide(currentSlide);
    startAutoPlay();
    
    // Pause auto-play on hover
    const slider = document.querySelector('.image-slider');
    if (slider) {
        slider.addEventListener('mouseenter', () => {
            isHovered = true;
            stopAutoPlay();
        });
        
        slider.addEventListener('mouseleave', () => {
            isHovered = false;
            startAutoPlay();
        });
    }
    
    // Touch/swipe support for mobile
    let startX = 0;
    let endX = 0;
    
    slider.addEventListener('touchstart', (e) => {
        startX = e.touches[0].clientX;
    });
    
    slider.addEventListener('touchend', (e) => {
        endX = e.changedTouches[0].clientX;
        handleSwipe();
    });
    
    function handleSwipe() {
        const threshold = 50;
        const diff = startX - endX;
        
        if (Math.abs(diff) > threshold) {
            if (diff > 0) {
                moveSlide(1); // Swipe left - next slide
            } else {
                moveSlide(-1); // Swipe right - previous slide
            }
        }
    }
});
</script>

<!-- <hr>

<h3>Tech Socials</h3>

<div class="socials-container">
    <div class="socials-grid">
        <a href="https://x.com/shrikantnaiidu" target="_blank" rel="noreferrer" class="social-link">
            <img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/socials/twitter.svg" alt="X" />
        </a>
        <a href="https://www.linkedin.com/in/shrikant-naidu/" target="_blank" rel="noreferrer" class="social-link">
            <img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/socials/linkedin.svg" alt="LinkedIn" />
        </a>
        <a href="https://github.com/shrikantnaidu" target="_blank" rel="noreferrer" class="social-link">
            <img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/socials/github.svg" alt="GitHub" />
        </a>
        <a href="https://steamcommunity.com/id/shrikantnaidu/" target="_blank" rel="noreferrer" class="social-link">
            <img src="https://www.vectorlogo.zone/logos/steampowered/steampowered-icon.svg" alt="Steam" />
        </a>
        <a href="https://wandb.ai/skn97" target="_blank" rel="noreferrer" class="social-link">
            <img src="https://www.vectorlogo.zone/logos/wandbai/wandbai-official.svg" alt="Weights & Biases" />
        </a>
        <a href="https://huggingface.co/shrikantnaidu" target="_blank" rel="noreferrer" class="social-link">
            <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" />
        </a>
    </div>
</div>

<hr> -->