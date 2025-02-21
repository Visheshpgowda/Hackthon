import jwt from "jsonwebtoken"
import { User } from "../models/user.models.js"

export const verifyJWT = async(req, res, next) => {
    try {
        // Check both cookies and Authorization header for the token
        const token = req.cookies?.accessToken || req.header("Authorization")?.replace("Bearer ", "")
        
        if (!token) {
            return res.status(401).json({
                success: false,
                message: "Unauthorized request"
            })
        }
    
        // Verify the token
        const decodedToken = jwt.verify(token, process.env.ACCESS_TOKEN_SECRET)
    
        // Find user but exclude sensitive fields
        const user = await User.findById(decodedToken?._id).select("-password -refreshToken")
    
        if (!user) {
            return res.status(401).json({
                success: false,
                message: "Invalid Access Token"
            })
        }

        // Set user in request object
        req.user = user
        next()

    } catch (error) {
        return res.status(401).json({
            success: false,
            message: error?.message || "Invalid access token"
        })
    }
}