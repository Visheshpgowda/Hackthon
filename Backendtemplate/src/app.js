import express from "express"
import cors from "cors"
import cookieParser from 'cookie-parser';

const app = express()

app.use(cors({
    origin: [
        process.env.CORS_ORIGIN
    ],
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-CSRF-Token', 'X-Requested-With', 'Accept', 'Accept-Version', 'Content-Length', 'Content-MD5', 'Date', 'X-Api-Version']
}));

//common middlewares
app.use(express.json({ limit: "1000kb" }))
app.use(express.urlencoded({ extended: true, limit: "1000kb" }))
app.use(express.static("public"))
app.use(cookieParser());


import healthcheckRouter from "./routes/healthcheck.routes.js"
import userRouter from "./routes/user.routes.js"
app.use("/api/v1/healthcheck", healthcheckRouter)
app.use("/api/v1/users", userRouter);



export { app }