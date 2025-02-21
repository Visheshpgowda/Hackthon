import { asyncHandler } from "../../utils/asyncHandler.js";
import { User } from "../../models/user.models.js";
import { ApiResponse } from "../../utils/ApiResponse.js"


const registerUser = asyncHandler(async(req, res) => {
    const { name, email, Phone_no, password, role } = req.body;
    if (name == "" || email == "" || Phone_no == "" || password == "" || role == "") {
        return res.status(400).json({ message: "All fields are required" });
    }
    const userExist = await User.findOne({
        $or: [{ name }, { email }]
    });
    if (userExist) {
        return res.status(409).json({ message: "User already exists" });
    }
    const user = await User.create({
        name,
        email,
        Phone_no,
        password,
        role
    })
    const Createduser = await User.findById(user._id).select("-password -refreshToken");
    if (!Createduser) {
        return res.status(500).json({ message: "User not created" });
    }
    return res.status(201).json(
        new ApiResponse(201, Createduser, "User created successfully")
    )



})

export { registerUser }