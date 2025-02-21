import { asyncHandler } from "../../utils/asyncHandler.js";
import { User } from "../../models/user.models.js";
import { ApiResponse } from "../../utils/ApiResponse.js";
import { ApiError } from "../../utils/ApiError.js";

const generateAccessTokenandRefreshToken = async (userId) => {
    try {
        console.log("came here at generation of tojkens method");
        console.log(userId);
        
        const user = await User.findById(userId); 
        
        if (!user) throw new ApiError(404, "User not found");

        const accessToken = await user.generateAccessToken();
        const refreshToken = await user.generateRefreshToken(); 
        console.log(`access are ${accessToken}`);
        
        user.refreshToken = refreshToken;
        await user.save({ validateBeforeSave: false });
        console.log(accessToken, refreshToken);
        
        return { accessToken, refreshToken };
    } catch (error) {
        console.log("some error", error);
        
    }
};

const loginUser = asyncHandler(async (req, res) => {
    const { email, name, password } = req.body;

    if (!email || !password || !name) {
        return res.status(400).json({ message: "Email, name, and password are required" });
    }

    const user = await User.findOne({ $or: [{ email }, { name }] });
    console.log(user);
    
    if (!user) {
        return res.status(404).json({ message: "User not found" });
    }

    const isPasswordCorrect = await user.isPasswordcorrect(password);
    if (!isPasswordCorrect) {
        return res.status(401).json({ message: "Invalid password" });
    }
    console.log("password crrct");
    
    // FIX: Await token generation function
    const { accessToken, refreshToken } = await generateAccessTokenandRefreshToken(user._id);
    console.log("token gen done");
    

    // FIX: Await the user query
    const loggedInUser = await User.findById(user._id).select("-password -refreshToken");

    const options = {
        httpOnly: true,
        secure: true
    };

    return res
        .status(200)
        .cookie("accessToken", accessToken, options)
        .cookie("refreshToken", refreshToken, options)
        .json(new ApiResponse(200, { user: loggedInUser }, "User logged in successfully"));
});

const logoutUser = asyncHandler(async (req, res) => {
    await User.findByIdAndUpdate(
        req.user._id,
        { $unset: { refreshToken: 1 } }, // Removes refreshToken from DB
        { new: true }
    );

    const options = {
        httpOnly: true,
        secure: true
    };

    return res
        .status(200)
        .clearCookie("accessToken", options)
        .clearCookie("refreshToken", options)
        .json(new ApiResponse(200, {}, "User logged out successfully"));
});

export { loginUser, logoutUser };
