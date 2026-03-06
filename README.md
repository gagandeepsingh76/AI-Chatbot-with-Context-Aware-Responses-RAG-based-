# 📊 CRM Dashboard

## 📌 Description

The **CRM Dashboard** is a web application that provides a comprehensive overview of key metrics for customer relationship management. It includes visualizations such as line charts and pie charts to display user data and trends over various time ranges.

This application is built using **React.js** for the frontend, along with **Chart.js** for rendering charts, and **Axios** for data fetching.

---

# 🚀 Features

- **Dynamic Data Fetching**  
  Fetches and updates data based on selected time ranges.

- **User Metrics**  
  Displays total users, new leads, closed deals, and open tickets.

- **Charts**  
  Includes pie charts and line charts to visualize data.

- **Responsive Design**  
  Designed to be fully responsive and accessible on various devices.

---

# 🛠 Technologies Used

### Frontend
- React.js
- Tailwind CSS

### Charts
- Chart.js
- React Chart.js 2

### Data Fetching
- Axios

### Styling
- Tailwind CSS

---

# ⚙ Getting Started

## Prerequisites

- Node.js (v14 or later)
- npm (v6 or later) or yarn

---

# 📥 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/crm-dashboard.git
```

### 2️⃣ Change Directory

```bash
cd crm-dashboard
```

### 3️⃣ Install Dependencies

```bash
npm install
```

---

# ▶ Running the Application

### Start the Development Server

```bash
npm run dev
```

---

# 🧑‍💻 Usage

### 1️⃣ Select Time Range

The `TimeSelector` component allows users to select a time range for the dashboard data.

Available options:

- 1 Day
- 1 Week
- 1 Month
- 1 Year

When a time range is selected, the dashboard dynamically fetches and displays data corresponding to that period.

---

### 2️⃣ View Metrics

The dashboard displays the following key metrics in the form of cards:

- **Total Users**  
  The total number of active, inactive, and new users.

- **New Leads**  
  The number of leads that have been converted over the selected time range.

- **Closed Deals**  
  The number of deals that were lost.

- **Open Tickets**  
  The number of unresolved tickets.

---

### 3️⃣ View Charts

#### Pie Chart

The `PieChart` component visually represents the distribution of users across different departments such as:

- Marketing
- Sales
- Support

Each department is color-coded to distinguish them easily.

---

#### Line Chart

The `LineChart` component shows the trend of leads over the selected time range.

It helps visualize how the number of leads fluctuates over time. Data points are plotted for each time period (e.g., monthly or yearly), allowing tracking of progress or changes effectively.

---

### 4️⃣ Data Fetching

The data is fetched from a mock API using **axios**.

The API returns the number of users, leads, and tickets for each time range. The data is dynamically updated based on the selected time range and displayed in the respective charts and cards.

---

# 📂 Project Structure

```
CRM-Dashboard
│
├── public
│   ├── tailor talk.svg
│   └── vite.svg
│
├── src
│   ├── Pages
│   │   └── Dashboard.jsx
│   │
│   ├── api
│   │   └── api.js
│   │
│   ├── assets
│   │   └── react.svg
│   │
│   ├── components
│   │   ├── Cards.jsx
│   │   ├── LineChart.jsx
│   │   ├── PieChart.jsx
│   │   └── TimeSelector.jsx
│   │
│   ├── services
│   │   └── dataService.js
│   │
│   ├── App.css
│   ├── App.jsx
│   ├── index.css
│   └── main.jsx
│
├── README.md
├── index.html
├── new.json
├── package.json
├── package-lock.json
├── eslint.config.js
├── postcss.config.js
├── tailwind.config.js
└── vite.config.js
```

---

# 📸 Dashboard Preview

<p align="center">

<img width="910" height="623" alt="CRM Dashboard Preview" src="https://github.com/user-attachments/assets/44386b7a-4b15-4a62-a27b-1fe578923628" />

</p>

---

# 👨‍💻 Author

**Gagandeep Singh**

Computer Science Student  
Interested in **Artificial Intelligence, Web Development, and Data Visualization**

---

# ⭐ GitHub Repository Description

CRM Dashboard built using **React.js, Tailwind CSS, Chart.js, and Axios** to visualize customer relationship management metrics with interactive charts and dynamic data fetching.

---

# 🏷 GitHub Topics

```
react
crm-dashboard
chartjs
data-visualization
tailwindcss
axios
dashboard-ui
web-application
frontend
```

---

⭐ If you like this project, consider giving it a **star** on GitHub!
