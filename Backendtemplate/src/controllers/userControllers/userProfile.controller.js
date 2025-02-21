import { asyncHandler } from "../../utils/asyncHandler.js";
import { User } from "../../models/user.models.js";
import { ApiResponse } from "../../utils/ApiResponse.js";

// This endpoint fetches the logged-in user's profile
const getProfile = asyncHandler(async (req, res) => {
    // Ensure that the user is authenticated (middleware should have set req.user)
    if (!req.user) {
        return res.status(401).json({ message: "Unauthorized" });
    }
    const user = await User.findById(req.user._id).select("-password -refreshToken");
    if (!user) {
        return res.status(404).json({ message: "User not found" });
    }
    return res.status(200).json(
        new ApiResponse(200, { user }, "Profile fetched successfully")
    );
});

export { getProfile };
