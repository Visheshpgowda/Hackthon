import express from "express"
import cors from "cors"
import cookieParser from 'cookie-parser';

const app = express()
console.log(process);

app.use(cors({
    origin: 'http://localhost:5173',  // Explicitly set the origin instead of using env variable for testing
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
    allowedHeaders: ['Content-Type', 'Authorization']
})); 

//common middlewares
app.use(express.json({ limit: "1000kb" }))
app.use(express.urlencoded({ extended: true, limit: "1000kb" }))
app.use(express.static("public"))
app.use(cookieParser());


import healthcheckRouter from "./routes/healthcheck.routes.js"
import userRouter from "./routes/user.routes.js"
import studentRouter from "./routes/student.routes.js"
app.use("/api/v1/healthcheck", healthcheckRouter)
app.use("/api/v1/users", userRouter);
app.use("/api/v1/students", studentRouter);




export { app }