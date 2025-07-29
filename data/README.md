# Customer Case Management System

A modern web application built with Next.js 14 for managing customer support cases, featuring separate interfaces for customers and analysts.

## Features

- 🔐 Secure authentication for both customers and analysts
- 📝 Case creation and management
- 📁 Document upload and management
- 💬 Real-time case updates
- 📧 Email integration for customer communication
- 🎯 Case categorization and priority management
- 📊 Dynamic form fields based on case category
- 🔍 Case filtering and search capabilities

## Prerequisites

Before you begin, ensure you have installed:
- [Node.js](https://nodejs.org/) (v18.17 or higher)
- [npm](https://www.npmjs.com/) (v9.0 or higher)
- [Git](https://git-scm.com/) for version control

## Setup Instructions

### 1. Clone the Repository

```bash
git clone [repository-url]
cd customer-case-management
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Environment Setup

Create a `.env` file in the root directory with the following variables:

```env
# Database
DATABASE_URL="file:./dev.db"

# Authentication
JWT_SECRET="your-jwt-secret-key"
NEXTAUTH_SECRET="your-nextauth-secret"
NEXTAUTH_URL="http://localhost:3000"

# File Upload
UPLOAD_DIR="uploads"
```

Replace the secret keys with secure random strings.

### 4. Database Setup

Initialize the database and run migrations:

```bash
npx prisma generate
npx prisma db push
```

### 5. Create Upload Directory

Create a directory for file uploads:

```bash
mkdir uploads
```

### 6. Run Development Server

```bash
npm run dev
```

The application will be available at `http://localhost:3000`

## Project Structure

```
customer-case-management/
├── app/                    # Next.js app directory
│   ├── api/               # API routes
│   ├── analyst/           # Analyst pages
│   ├── components/        # Shared components
│   ├── lib/              # Utility functions
│   └── page.tsx          # Home page
├── prisma/               # Database schema and migrations
├── public/              # Static assets
└── uploads/             # File upload directory
```

## User Types and Access

### 1. Customers
- Can create new cases
- Upload documents
- View their case history
- Receive email updates

### 2. Analysts
- View and manage all cases
- Update case status and priority
- Add case notes
- Download and view documents
- Send email updates to customers

## API Routes

- `/api/auth/*` - Authentication endpoints
- `/api/cases` - Case management
- `/api/cases/[id]` - Individual case operations
- `/api/cases/[id]/updates` - Case updates
- `/api/cases/[id]/documents/[documentId]` - Document operations

## Technology Stack

- **Frontend**: Next.js 14, React, TypeScript, TailwindCSS
- **UI Components**: shadcn/ui
- **Backend**: Next.js API Routes
- **Database**: SQLite with Prisma ORM
- **Authentication**: JWT with HTTP-only cookies
- **File Storage**: Local filesystem
- **Form Handling**: React Hook Form
- **Notifications**: Sonner
- **Icons**: Lucide React
- **Date Handling**: date-fns

## Development Guidelines

1. **Code Style**
   - Use TypeScript for type safety
   - Follow ESLint rules
   - Use proper component organization
   - Implement error handling

2. **Security**
   - Validate all user inputs
   - Use HTTP-only cookies for auth
   - Implement proper file upload validation
   - Follow OWASP security guidelines

3. **Performance**
   - Optimize image and file uploads
   - Use proper caching strategies
   - Implement pagination where needed
   - Optimize API responses

## Common Issues and Solutions

1. **Database Connection**
   - Ensure SQLite file has proper permissions
   - Check DATABASE_URL in .env
   - Run latest migrations

2. **File Uploads**
   - Verify upload directory exists and is writable
   - Check file size limits
   - Ensure proper MIME type validation

3. **Authentication**
   - Clear cookies if experiencing login issues
   - Verify JWT_SECRET is set
   - Check NEXTAUTH_URL matches your environment

## Testing

Run the test suite:

```bash
npm run test
```

## Building for Production

```bash
npm run build
npm start
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.
