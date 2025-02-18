import mongoose,{Schema} from "mongoose";

const userSchema = new Schema({
    name:{
        type: String,
        required: true,
        trim: true
    },
    email: {
        type: String,
        required: true,
        unique: true,
    },
    Phone_no:{
        type: String,
        required: true,
        unique: true,
    },
    password:{
        type: String,
        required: true,
    },
    role:{
        type: String,
        enum: ["Headmaster","Teacher","Parent"],
        default: "user",
    },
})

export const User = mongoose.model("User", userSchema)