Write-Host "Starting frontend development server..."
Write-Host "The server will be available at http://localhost:3000"
Write-Host "Press Ctrl+C to stop the server."

# Change to frontend directory
Set-Location -Path ".\frontend"

# Run development server
npm run dev

# Return to original directory when done
Set-Location -Path ".."