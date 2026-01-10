# Clothing Reconstruction Frontend

A modern React frontend application for the Clothing Reconstruction API.

## Features

- 🖼️ **Image Upload**: Drag & drop or click to upload images
- 📊 **Real-time Status**: Server health monitoring
- 🎨 **Modern UI**: Beautiful, responsive design
- 📦 **Results Display**: Preview and download reconstructed clothing items
- ⚡ **Fast & Responsive**: Built with Vite and React

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Flask backend running on `http://localhost:5000`

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:3000`

## Building for Production

To create a production build:
```bash
npm run build
```

The built files will be in the `dist` directory.

## API Integration

The frontend communicates with the Flask backend API:

- **Health Check**: `GET http://localhost:5000/health`
- **Reconstruct**: `POST http://localhost:5000/reconstruct`

The API endpoint expects a multipart/form-data request with an image file.

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ImageUpload.tsx      # Image upload component
│   │   ├── ResultsDisplay.tsx   # Results display component
│   │   └── HealthStatus.tsx     # Server health status
│   ├── App.tsx                  # Main application component
│   ├── App.css                  # Application styles
│   ├── main.tsx                 # Application entry point
│   └── index.css                # Global styles
├── public/                       # Static assets
├── index.html                    # HTML template
├── vite.config.ts               # Vite configuration
└── package.json                 # Dependencies

```

## Technologies Used

- **React** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **JSZip** - ZIP file handling
- **CSS3** - Styling

## Notes

- Make sure the Flask backend is running before using the frontend
- The frontend expects the backend to be running on `http://localhost:5000`
- CORS must be enabled on the Flask backend for the frontend to work properly

