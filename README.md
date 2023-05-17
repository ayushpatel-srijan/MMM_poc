1. Connect using PEM Key:
   - Open your terminal or command prompt.
   - Use the following command to connect to the server using the PEM key:
     ```bash
     ssh -i path/to/mmm_poc_keypair.pem username@server_ip_address
     ```
   - Replace `path/to/mmm_poc_keypair.pem` with the actual path to your PEM key file.
   - Replace `username` with your server username.
   - Replace `server_ip_address` with the IP address of your server.

2. Navigate to MMM_poc Folder:
   - Once connected to the server, use the following command to navigate to the MMM_poc folder:
     ```bash
     cd MMM_poc
     ```

3. Run Docker Compose:
   - After navigating to the MMM_poc folder, run the following command to start the Docker Compose process:
     ```bash
     sudo docker-compose up
     ```
   - This command will start the Docker containers and deploy the MMM_poc application.
