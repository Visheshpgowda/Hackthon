import { asyncHandler } from "../../utils/asyncHandler"
import { User } from "../../models/user.models";
import {ApiResponse} from "../../utils/ApiResponse"
import { ApiError } from "../../utils/ApiError";

const generateAccessTokenandRefreshToken=async (userId)=>{
   try{
      const user=await User.findById(userId);
      const accessToken=await user.generateAccessToken();
      const refreshToken=await user.generateRefreshToken();
      user.refreshToken=refreshToken;
      await user.save({validateBeforeSave:false});
      return {accessToken,refreshToken};
   }catch(error){
         throw new ApiError(500,"something went wrong while generating token")
   }
}
const loginUser=asyncHandler(async(req,res)=>{
    const {email,name,password}=req.body;
    if(!email || !password  || !name){

        return res.status(400).json({message:"Email and password are required"});
    }
    const user=await User.findOne({
        $or:[{email},{name}]
    });

    if(!user){
        return res.status(404).json({message:"User not found"});
    }

    const isPasswordcorrect=await user.isPasswordcorrect(password);
    if(!isPasswordcorrect){
        return res.status(401).json({message:"Invalid password"});
    }

    const {accessToken,refreshToken}=generateAccessTokenandRefreshToken(user._id)

    const loggedinuser=User.findById(user._id).select("-password -refreshToken")
    const options={
        httpOnly:true,
        secure:true
    }


    return res
    .status(200)
    .cookie("accessToken",accessToken,options)
    .cookie("refreshToken",refreshToken,options)
    .json(
        new ApiResponse(200,"User logged in successfully",{user:loggedinuser})
    )



})

const logoutUser = asyncHandler(async(req, res) => {
    await User.findByIdAndUpdate(
        req.user._id,
        {
            $unset: {
                refreshToken: 1 // this removes the field from document
            }
        },
        {
            new: true
        }
    )

    const options = {
        httpOnly: true,
        secure: true
    }

    return res
    .status(200)
    .clearCookie("accessToken", options)
    .clearCookie("refreshToken", options)
    .json(new ApiResponse(200, {}, "User logged Out"))
})


// const logoutUser=asyncHandler(async(req,res)=>{
//     const {accessToken,refreshToken}=req.cookies;
//     if(!accessToken || !refreshToken){
//         return res.status(400).json({message:"User not logged in"})
//     }

//     const user=await  User.findById(req.user._id).select("-password -refreshToken");
export {loginUser,logoutUser}