import express from "express"
import cors from "cors"
import cookieParser from 'cookie-parser';

const app = express()

app.use(cors({
    origin: 'http://localhost:5173', // Specify the exact frontend URL
    credentials: true, // Allow cookies and authorization headers
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
    allowedHeaders: ['Content-Type', 'Authorization'],
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